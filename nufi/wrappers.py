import pandas as pd
import numpy as np
from nufi.impute import NufiImputer

def infill_dataframe(df, imputer=None, time_col=None, keep_time_col=False):
    """
    Infill a standard single-index or column-based Pandas / cuDF DataFrame.
    If cuDF is detected, handles GPU memory transfer seamlessly.

    .. warning::
        The imputer instance is **not thread-safe**. If the same ``NufiImputer``
        object is shared across threads or concurrent calls, internal fitted
        state (timestamps, GCV parameters, LDL^T factors) will race and may
        produce corrupted results. Use a separate instance per thread or
        protect calls with external locking.

    Parameters:
    -----------
    df : pandas.DataFrame or cudf.DataFrame
        The DataFrame to infill.
    imputer : NufiImputer, optional
        The imputer instance to use. If None, a new NufiImputer is created.
    time_col : str, optional
        The column name to use as timestamps. If provided, this column is set as index.
    keep_time_col : bool, default False
        Whether to keep the time column as a feature in the DataFrame columns
        passed to the imputer. If False, the time column is set as the index and
        removed from the features.
        Note: When keep_time_col=True, the time column is duplicated as both the DataFrame index
        and a regular feature column. The imputer will fit/transform it like any other column,
        which may distort covariance estimation if timestamp values differ in scale.
    """
    try:
        import cudf
        is_cudf = isinstance(df, cudf.DataFrame)
    except ImportError:
        is_cudf = False

    if is_cudf:
        # Convert cuDF to Pandas for sklearn compatibility
        pd_df = df.to_pandas()
    else:
        pd_df = df.copy()
        
    if imputer is None:
        imputer = NufiImputer()
        
    # If a specific column is defined as time, set it as index
    if time_col is not None:
        if time_col not in pd_df.columns:
            raise ValueError(
                f"time_col '{time_col}' not found in DataFrame columns: {list(pd_df.columns)}"
            )
        # Capture the original index name before replacement and warn if discarded
        previous_index_name = pd_df.index.name
        if previous_index_name is not None:
            import warnings
            warnings.warn(
                f"DataFrame index name {previous_index_name!r} is being discarded "
                "as it is being replaced by time_col.",
                UserWarning
            )
        if keep_time_col:
            import warnings
            warnings.warn(
                "keep_time_col=True duplicates timestamps as both index and feature. "
                "Timestamp magnitudes (e.g., Unix nanoseconds) may dominate covariance "
                "estimation and produce biased imputations for other columns. "
                "Consider normalizing timestamps or using keep_time_col=False.",
                UserWarning
            )
            time_values = pd_df[time_col].copy()
            col_pos = pd_df.columns.get_loc(time_col)
            pd_df = pd_df.set_index(time_col)
            pd_df.index.name = None  # avoid name collision with the column
            pd_df.insert(col_pos, time_col, time_values)
            # Note: the time column is now both the index and a feature column.
            # The imputer will fit/transform it like any other column, which may
            # distort covariance estimation if timestamp values differ in scale.
        else:
            pd_df = pd_df.set_index(time_col)
            pd_df.index.name = None
        
    infilled_pd = imputer.fit_transform(pd_df)
    
    if len(infilled_pd) != len(pd_df):
        raise ValueError(
            f"Imputer returned {len(infilled_pd)} rows, "
            f"expected {len(pd_df)}. "
            "The NufiImputer must preserve row count."
        )
    
    if is_cudf:
        return cudf.DataFrame.from_pandas(infilled_pd)
    return infilled_pd

def infill_multiindex_dataframe(df, imputer=None, entity_level=0, time_level=1, sort=True):
    """
    Infill a MultiIndex Pandas/cuDF DataFrame (typically panel data).
    Each entity group (e.g. per-entity time series) is infilled independently
    to preserve distinct group behaviors and covariance.

    .. warning::
        The imputer instance is **not thread-safe**. If the same ``NufiImputer``
        object is shared across threads or concurrent calls, internal fitted
        state (timestamps, GCV parameters, LDL^T factors) will race and may
        produce corrupted results. Use a separate instance per thread or
        protect calls with external locking.

    Parameters:
    -----------
    df : pandas.DataFrame or cudf.DataFrame
        The MultiIndex DataFrame to infill.
    imputer : NufiImputer, optional
        The imputer instance to use. If None, a new NufiImputer is created.
    entity_level : int or str, default 0
        The level of the MultiIndex representing distinct entities/groups.
    time_level : int or str, default 1
        The level of the MultiIndex representing timestamps.
    sort : bool, default True
        Whether to sort the index by time level to ensure proper chronological order.
        Setting sort=False when timestamps are not already sorted may produce
        incorrect results due to Nyquist frequency miscalculation.
        Note: This reorders the group's rows; the output index order will reflect the sorted
        order, not the original input order.
    """
    try:
        import cudf
        is_cudf = isinstance(df, cudf.DataFrame)
    except ImportError:
        is_cudf = False

    if is_cudf:
        pd_df = df.to_pandas()
    else:
        pd_df = df.copy()
        
    # Validate MultiIndex levels
    n_levels = pd_df.index.nlevels
    if isinstance(entity_level, int) and entity_level >= n_levels:
        raise ValueError(
            f"entity_level={entity_level} exceeds MultiIndex nlevels={n_levels}"
        )
    if isinstance(time_level, int) and time_level >= n_levels:
        raise ValueError(
            f"time_level={time_level} exceeds MultiIndex nlevels={n_levels}"
        )
        
    if imputer is None:
        imputer = NufiImputer()
        
    # Group by the entity level and apply NufiImputer
    # sort=False preserves original entity order; time ordering is handled separately
    grouped = pd_df.groupby(level=entity_level, group_keys=False, sort=False)
    
    def imputer_apply(group):
        # We need to sort index by time level to ensure proper chronological order
        if sort:
            group_sorted = group.sort_index(level=time_level)
        else:
            group_sorted = group

        # Drop the multi-index temporarily for fit_transform but keep index values as timestamps
        timestamps = group_sorted.index.get_level_values(time_level).to_numpy()
        
        # Validate timestamp convertibility early
        try:
            np.array(timestamps, dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(
                f"Timestamp level '{time_level}' has dtype {timestamps.dtype}, "
                f"which cannot be converted to float64. Use a numeric or datetime64 index level."
            )

        # Validate strictly monotonic timestamps to prevent Nyquist overflow
        if len(timestamps) > 1:
            diffs = np.diff(timestamps.astype(np.float64))
            if np.any(diffs <= 0):
                raise ValueError(
                    f"Timestamps for group must be strictly increasing; "
                    f"found non-positive or zero difference. "
                    f"Check for duplicate or out-of-order timestamps."
                )

        # Avoid to_numpy() which coerces dtypes; copy with a clean index instead
        temp_df = group_sorted.copy()
        temp_df.index = timestamps
        
        try:
            infilled_temp = imputer.fit_transform(temp_df)
        except Exception as e:
            entity_id = group_sorted.index.get_level_values(entity_level)[0]
            raise RuntimeError(
                f"NufiImputer.fit_transform failed for entity {entity_id!r}: {e}"
            ) from e
        
        if len(infilled_temp) != len(group_sorted):
            raise ValueError(
                f"Imputer returned {len(infilled_temp)} rows, "
                f"expected {len(group_sorted)}. "
                "The NufiImputer must preserve row count."
            )
        
        # Restore MultiIndex structure
        infilled_temp.index = group_sorted.index
        return infilled_temp
        
    # Avoid groupby.apply double-call on first group by iterating manually
    infilled_dfs = []
    for _, group in grouped:
        infilled_dfs.append(imputer_apply(group))
    infilled_pd = pd.concat(infilled_dfs)
    
    if len(infilled_pd) != len(pd_df):
        raise ValueError(
            f"Concatenated result has {len(infilled_pd)} rows, "
            f"expected {len(pd_df)}. Group-level indices may overlap or be non-unique."
        )
    
    if is_cudf:
        return cudf.DataFrame.from_pandas(infilled_pd)
    return infilled_pd
