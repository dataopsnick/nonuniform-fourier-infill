import pandas as pd
import numpy as np
from nufi.impute import NufiImputer

def infill_dataframe(df, imputer=None, time_col=None):
    """
    Infill a standard single-index or column-based Pandas / cuDF DataFrame.
    If cuDF is detected, handles GPU memory transfer seamlessly.
    """
    is_cudf = False
    if type(df).__name__ == "DataFrame" and type(df).__module__.startswith("cudf"):
        is_cudf = True
        # Convert cuDF to Pandas for sklearn compatibility
        pd_df = df.to_pandas()
    else:
        pd_df = df.copy()
        
    if imputer is None:
        imputer = NufiImputer()
        
    # If a specific column is defined as time, set it as index
    if time_col is not None:
        pd_df = pd_df.set_index(time_col)
        
    infilled_pd = imputer.fit_transform(pd_df)
    
    if is_cudf:
        import cudf
        return cudf.DataFrame.from_pandas(infilled_pd)
    return infilled_pd

def infill_multiindex_dataframe(df, imputer=None, entity_level=0, time_level=1):
    """
    Infill a MultiIndex Pandas/cuDF DataFrame (typically panel data).
    Each entity group (e.g. per-entity time series) is infilled independently
    to preserve distinct group behaviors and covariance.
    """
    is_cudf = False
    if type(df).__name__ == "DataFrame" and type(df).__module__.startswith("cudf"):
        is_cudf = True
        pd_df = df.to_pandas()
    else:
        pd_df = df.copy()
        
    if imputer is None:
        imputer = NufiImputer()
        
    # Group by the entity level and apply NufiImputer
    grouped = pd_df.groupby(level=entity_level, group_keys=False)
    
    def imputer_apply(group):
        # We need to sort index by time level to ensure proper chronological order
        group_sorted = group.sort_index(level=time_level)
        # Drop the multi-index temporarily for fit_transform but keep index values as timestamps
        timestamps = group_sorted.index.get_level_values(time_level).to_numpy()
        # Create a single index df
        temp_df = pd.DataFrame(group_sorted.to_numpy(), index=timestamps, columns=group_sorted.columns)
        infilled_temp = imputer.fit_transform(temp_df)
        
        # Restore MultiIndex structure
        infilled_temp.index = group_sorted.index
        return infilled_temp
        
    infilled_pd = grouped.apply(imputer_apply)
    
    if is_cudf:
        import cudf
        return cudf.DataFrame.from_pandas(infilled_pd)
    return infilled_pd
