import pandas as pd
import numpy as np
from nufi.impute import NufiImputer, evaluate_fourier_reconstruction
from sklearn.utils import check_random_state

def infill_dataframe(df, imputer=None, time_col=None, keep_time_col=False, sort=True):
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
    sort : bool, default True
        Whether to sort by the time column or index before calling fit_transform.
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

    if time_col is not None and time_col not in pd_df.columns:
        raise ValueError(
            f"time_col '{time_col}' not found in DataFrame columns: {list(pd_df.columns)}"
        )
        
    if imputer is None:
        imputer = NufiImputer()
        
    # Sort by time to ensure proper chronological order for frequency estimation
    if sort:
        if time_col is not None:
            pd_df = pd_df.sort_values(time_col)
        else:
            pd_df = pd_df.sort_index()
    else:
        import warnings
        warnings.warn(
            "sort=False: timestamps will not be sorted. "
            "Non-monotonic timestamps may produce incorrect frequency estimates.",
            UserWarning
        )

    # If a specific column is defined as time, set it as index
    if time_col is not None:
        if pd_df[time_col].isna().any():
            raise ValueError(
                f"time_col '{time_col}' contains NaN values, which are not valid as timestamps."
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
        
    # Validate timestamp integrity (parity with multiindex wrapper)
    timestamps = pd_df.index.values
    try:
        np.array(timestamps, dtype=np.float64)
    except (TypeError, ValueError):
        raise TypeError(
            f"Index dtype {pd_df.index.dtype} cannot be converted to float64. "
            "Use a numeric or datetime64 index."
        )
    if len(timestamps) > 1:
        if np.issubdtype(timestamps.dtype, np.datetime64):
            if pd.isna(timestamps).any():
                raise ValueError("Index contains NaT values.")
            diffs = np.diff(timestamps.view('int64'))
        else:
            if np.isnan(timestamps.astype(np.float64)).any():
                raise ValueError("Index contains NaN values.")
            diffs = np.diff(timestamps.astype(np.float64))
        if np.any(diffs <= 0):
            raise ValueError(
                "Timestamps must be strictly increasing; "
                "found non-positive difference."
            )

    infilled_pd = imputer.fit_transform(pd_df)
    
    if len(infilled_pd) != len(pd_df):
        raise ValueError(
            f"Imputer returned {len(infilled_pd)} rows, "
            f"expected {len(pd_df)}. "
            "The NufiImputer must preserve row count."
        )
    
    # Verify row order before continuing
    if not infilled_pd.index.equals(pd_df.index):
        raise ValueError(
            "NufiImputer.fit_transform reordered rows. "
            "Row order must be preserved."
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
            import warnings
            warnings.warn(
                "sort=False: timestamps will not be sorted. "
                "Non-monotonic timestamps may produce incorrect frequency estimates.",
                UserWarning
            )
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
            # Use integer diff on datetime64 to avoid float64 precision loss
            if np.issubdtype(timestamps.dtype, np.datetime64):
                if pd.isna(timestamps).any():
                    raise ValueError("Timestamp level contains NaT values.")
                diffs = np.diff(timestamps.view('int64'))
            else:
                if np.isnan(timestamps.astype(np.float64)).any():
                    raise ValueError("Timestamp level contains NaN values.")
                diffs = np.diff(timestamps.astype(np.float64))
            if np.any(diffs <= 0):
                raise ValueError(
                    "Timestamps must be strictly increasing; "
                    "found non-positive difference."
                )

        # Avoid to_numpy() which coerces dtypes; copy with a clean index instead
        temp_df = group_sorted.copy()
        temp_df.index = timestamps
        
        # Use a fresh imputer per group to avoid cross-group contamination
        group_imputer = NufiImputer() if imputer is None else imputer.clone()
        
        try:
            infilled_temp = group_imputer.fit_transform(temp_df)
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
        
        # Verify row order before restoring index
        if not infilled_temp.index.equals(temp_df.index):
            raise ValueError(
                "NufiImputer.fit_transform reordered rows. "
                "Row order must be preserved for correct index restoration."
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

    if not infilled_pd.index.equals(pd_df.index):
        raise ValueError(
            "Concatenated result index differs from input index. "
            "Group-level indices may overlap or be non-unique."
        )

    if is_cudf:
        return cudf.DataFrame.from_pandas(infilled_pd)
    return infilled_pd


def infill_rbmi_dataframe(
    df,
    arm_col,
    reference_arm,
    subject_col=None,
    time_col=None,
    rbmi_method="J2R",
    imputer=None,
    stochastic=True,
    n_imputations=1,
    random_state=None,
):
    """
    Infill a clinical trial longitudinal DataFrame using Reference-Based
    Multiple Imputation (RBMI; Carpenter, Roger & Kenward, 2013).

    Supported mechanisms for active-arm subjects with dropout time t_d:
    - "CR"  (Copy Reference): every missing value uses the reference arm
      NUDFT mean trajectory plus reference LDL^T posterior noise.
    - "J2R" (Jump to Reference): pre-dropout follows the subject own NUDFT
      reconstruction; post-dropout (t > t_d) jumps to the reference mean
      trajectory plus reference noise.
    - "CIR" (Copy Increments in Reference): post-dropout anchors at the
      subject last observed value x(t_d) and accumulates the reference
      Fourier increments x_ref(t) - x_ref(t_d) (the exact integral of
      the analytical reference derivative), preserving C^0 continuity at
      t_d and C^infinity smoothness afterwards.
    - "LMCF" (Last Mean Carried Forward): post-dropout mean held constant
      at the subject value at t_d.

    Returns a single infilled DataFrame when n_imputations == 1, otherwise
    a list of M infilled DataFrames. Observed values are strictly invariant
    and reference-arm rows are returned unchanged.
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

    if not isinstance(pd_df, pd.DataFrame):
        raise TypeError("df must be a pandas or cuDF DataFrame")

    if rbmi_method not in ("CR", "J2R", "CIR", "LMCF"):
        raise ValueError(
            "Unsupported rbmi_method %r. Expected one of: "
            "CR, J2R, CIR, LMCF." % (rbmi_method,)
        )
    if not isinstance(n_imputations, (int, np.integer)) or n_imputations < 1:
        raise ValueError(
            "n_imputations must be a positive integer, got %r."
            % (n_imputations,)
        )
    if arm_col not in pd_df.columns:
        raise KeyError(
            "arm_col %r not found in DataFrame columns: %s"
            % (arm_col, list(pd_df.columns))
        )
    if time_col is not None and time_col not in pd_df.columns:
        raise KeyError(
            "time_col %r not found in DataFrame columns: %s"
            % (time_col, list(pd_df.columns))
        )
    if subject_col is not None and subject_col not in pd_df.columns:
        raise KeyError(
            "subject_col %r not found in DataFrame columns: %s"
            % (subject_col, list(pd_df.columns))
        )
    if reference_arm not in set(pd.unique(pd_df[arm_col])):
        raise ValueError(
            "reference_arm %r not found in arm column %r."
            % (reference_arm, arm_col)
        )

    if time_col is not None:
        times_num = pd.to_numeric(pd_df[time_col], errors="coerce")
        if times_num.isna().any():
            raise ValueError(
                "time_col %r contains non-numeric / NaN values." % (time_col,)
            )
        times = times_num.to_numpy(dtype=np.float64)
    else:
        try:
            times = pd_df.index.to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError(
                "DataFrame index must be numeric when time_col is None."
            )
        if np.isnan(times).any():
            raise ValueError(
                "Index contains NaN timestamps; provide time_col or "
                "clean the index."
            )

    if subject_col is not None:
        subject_ids = pd_df[subject_col].to_numpy()
    else:
        # No subject column: each row is its own independent series.
        subject_ids = np.arange(len(pd_df))

    arms = pd_df[arm_col].to_numpy()
    exclude = {arm_col}
    if subject_col is not None:
        exclude.add(subject_col)
    if time_col is not None:
        exclude.add(time_col)
    outcome_cols = [
        c for c in pd_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(pd_df[c])
    ]
    if not outcome_cols:
        raise ValueError("No numeric outcome columns found to infill.")

    n_outcomes = len(outcome_cols)
    rng = check_random_state(random_state)
    if isinstance(imputer, NufiImputer):
        template = imputer
    else:
        template = NufiImputer()

    unique_subjects = pd.unique(subject_ids)
    ref_subjects = []
    active_subjects = []
    for s in unique_subjects:
        s_rows = np.where(subject_ids == s)[0]
        arm_label = arms[s_rows[0]]
        if arm_label == reference_arm:
            ref_subjects.append(s)
        else:
            active_subjects.append(s)
    if not ref_subjects:
        raise ValueError("Reference arm %r has no subjects." % (reference_arm,))
    if not active_subjects:
        raise ValueError("No active-arm subjects to impute.")

    ref_row_mask = arms == reference_arm
    ref_df = pd.DataFrame(
        pd_df.loc[ref_row_mask, outcome_cols].to_numpy(dtype=np.float64),
        columns=outcome_cols,
    )
    ref_df["_time"] = times[ref_row_mask]
    ref_mean_df = ref_df.groupby("_time", sort=True)[outcome_cols].mean()

    # Fit the reference arm NUDFT mean trajectory + LDL^T posterior covariance.
    ref_imputer = template.clone().set_params(
        mnar_strategy=None, delta_adjustment=0.0
    )
    ref_imputer.fit(ref_mean_df)
    ref_imputer.transform(ref_mean_df, stochastic=False)
    ref_coeffs = ref_imputer.coefficients_
    ref_freqs = ref_imputer.frequencies_

    def _ref_mean_at(j, t_query):
        coeff = ref_coeffs.get(j)
        freq = ref_freqs.get(j)
        if coeff is None or freq is None:
            import warnings
            warnings.warn(
                "Reference arm has no observations for outcome %r; "
                "using a zero reference trajectory." % (outcome_cols[j],)
            )
            return np.zeros(np.asarray(t_query).shape, dtype=np.float64)
        return evaluate_fourier_reconstruction(coeff, freq, t_query)

    base_outcomes = pd_df[outcome_cols].to_numpy(dtype=np.float64)
    draws_arrays = [base_outcomes.copy() for _ in range(n_imputations)]

    for s in active_subjects:
        rows = np.where(subject_ids == s)[0]
        order = np.argsort(times[rows], kind="stable")
        rows_sorted = rows[order]
        sub_times = times[rows_sorted]
        sub_data = base_outcomes[rows_sorted].copy()

        sub_df = pd.DataFrame(
            sub_data, index=sub_times, columns=outcome_cols
        )
        sub_imputer = template.clone().set_params(
            mnar_strategy=None, delta_adjustment=0.0
        )
        sub_imputer.fit(sub_df)
        sub_imputer.transform(sub_df, stochastic=False)

        subj_recon = np.full_like(sub_data, np.nan)
        for j in range(n_outcomes):
            rec = sub_imputer.reconstructed_.get(j)
            if rec is not None:
                subj_recon[:, j] = rec
        for j in range(n_outcomes):
            need = np.isnan(subj_recon[:, j])
            if need.any():
                col_obs = sub_data[:, j]
                if np.isfinite(col_obs).any():
                    fallback = float(np.nanmean(col_obs))
                else:
                    fallback = 0.0
                subj_recon[need, j] = fallback

        if stochastic:
            noise_subj = sub_imputer.sample_posterior_noise(
                len(rows_sorted), int(n_imputations), rng=rng
            )
            noise_ref = ref_imputer.sample_posterior_noise(
                len(rows_sorted), int(n_imputations), rng=rng
            )
        else:
            noise_shape = (n_imputations, len(rows_sorted), n_outcomes)
            noise_subj = np.zeros(noise_shape)
            noise_ref = np.zeros(noise_shape)
        if noise_subj.ndim == 2:
            noise_subj = noise_subj[None, :, :]
        if noise_ref.ndim == 2:
            noise_ref = noise_ref[None, :, :]

        for j in range(n_outcomes):
            miss = np.isnan(sub_data[:, j])
            if not miss.any():
                continue
            obs = ~miss
            if obs.any():
                t_d = float(np.max(sub_times[obs]))
                last_pos = np.where(obs)[0][np.argmax(sub_times[obs])]
                anchor_val = float(sub_data[last_pos, j])
            else:
                # Entirely unobserved outcome: treat all values as
                # post-dropout with the earliest available anchor.
                t_d = float(np.min(sub_times)) - 1.0
                anchor_val = float(subj_recon[0, j])
            ref_traj = np.asarray(
                _ref_mean_at(j, sub_times), dtype=np.float64
            )
            ref_at_td = float(_ref_mean_at(j, np.array([t_d]))[0])

            for m in range(n_imputations):
                imputed = sub_data[:, j].copy()
                if rbmi_method == "CR":
                    mean_traj = ref_traj
                    noise_traj = noise_ref[m, :, j]
                else:
                    pre = sub_times <= t_d
                    post = ~pre
                    mean_traj = subj_recon[:, j].copy()
                    if rbmi_method == "J2R":
                        mean_traj[post] = ref_traj[post]
                    elif rbmi_method == "CIR":
                        mean_traj[post] = (
                            anchor_val + (ref_traj[post] - ref_at_td)
                        )
                    elif rbmi_method == "LMCF":
                        mean_traj[post] = anchor_val
                    noise_traj = np.where(
                        pre, noise_subj[m, :, j], noise_ref[m, :, j]
                    )
                imputed[miss] = mean_traj[miss] + noise_traj[miss]
                draws_arrays[m][rows_sorted, j] = imputed

    out_frames = []
    for m in range(n_imputations):
        frame = pd_df.copy()
        frame[outcome_cols] = draws_arrays[m]
        out_frames.append(frame)

    if n_imputations == 1:
        out = out_frames[0]
    else:
        out = out_frames
    if is_cudf:
        if n_imputations == 1:
            return cudf.DataFrame.from_pandas(out)
        return [cudf.DataFrame.from_pandas(f) for f in out]
    return out
