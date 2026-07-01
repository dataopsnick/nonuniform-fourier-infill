import numpy as np
import pandas as pd
import pytest
from nufi.impute import NufiImputer
from nufi.wrappers import infill_dataframe, infill_multiindex_dataframe

def test_infill_dataframe_wrapper():
    # Dedicated test for infill_dataframe (Task 18)
    data = {
        'timestamp': [1.0, 2.0, 3.0, 4.0],
        'signal': [10.0, np.nan, 30.0, np.nan]
    }
    df = pd.DataFrame(data)
    
    # Infill without keeping time_col as feature
    df_filled = infill_dataframe(df, time_col='timestamp', keep_time_col=False)
    assert isinstance(df_filled, pd.DataFrame)
    assert not df_filled.isna().any().any()
    assert len(df_filled) == len(df)
    
    # Infill keeping time_col as feature
    df_filled_keep = infill_dataframe(df, time_col='timestamp', keep_time_col=True)
    assert 'timestamp' in df_filled_keep.columns

def test_imputer_numpy():
    # Create simple data with NaNs
    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [3.0, np.nan],
        [4.0, 5.0]
    ], dtype=np.float64)
    
    imputer = NufiImputer(method='direct', covariance_compensation=False)
    X_filled = imputer.fit_transform(X)
    
    # Assert no NaNs remain
    assert not np.any(np.isnan(X_filled))
    assert X_filled.shape == X.shape
    # Check that non-nan values are preserved
    assert X_filled[0, 0] == 1.0
    assert X_filled[3, 1] == 5.0

def test_imputer_pandas():
    # DataFrame with NaNs
    data = {
        'signal_1': [10.0, np.nan, 30.0, 40.0],
        'signal_2': [20.0, 30.0, np.nan, 50.0]
    }
    df = pd.DataFrame(data, index=[1.1, 2.2, 3.3, 4.4])
    
    imputer = NufiImputer(method='fast', covariance_compensation=True)
    df_filled = imputer.fit_transform(df)
    
    assert isinstance(df_filled, pd.DataFrame)
    assert df_filled.shape == df.shape
    assert not df_filled.isna().any().any()
    assert df_filled.loc[1.1, 'signal_1'] == 10.0

def test_multiindex_wrapper():
    arrays = [
        ['group_A', 'group_A', 'group_B', 'group_B'],
        [1.0, 2.0, 1.0, 2.0]
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=('entity', 'time'))
    df = pd.DataFrame({'signal': [1.5, np.nan, np.nan, 3.5]}, index=index)
    
    imputer = NufiImputer(method='direct', covariance_compensation=False)
    df_filled = infill_multiindex_dataframe(df, imputer)
    
    assert isinstance(df_filled, pd.DataFrame)
    assert not df_filled.isna().any().any()
    assert df_filled.loc[('group_A', 1.0), 'signal'] == 1.5

def test_tikhonov_direct_and_cg_solvers():
    # Verify both direct and Conjugate Gradient (cg) solver types work
    X = np.array([
        [2.0, 4.0],
        [np.nan, 5.0],
        [6.0, np.nan],
        [8.0, 10.0]
    ], dtype=np.float64)
    
    imputer_direct = NufiImputer(method='direct', solver='direct', alpha=1e-3, covariance_compensation=False)
    X_direct = imputer_direct.fit_transform(X)
    
    imputer_cg = NufiImputer(method='direct', solver='cg', alpha=1e-3, covariance_compensation=False)
    X_cg = imputer_cg.fit_transform(X)
    
    assert not np.any(np.isnan(X_direct))
    assert not np.any(np.isnan(X_cg))
    assert X_direct.shape == X.shape
    assert X_cg.shape == X.shape
    # Check that non-nan values are preserved in both cases
    assert X_direct[0, 0] == 2.0
    assert X_cg[0, 0] == 2.0
    
    # Assert direct and CG solvers produce consistent/similar imputations (Task 20)
    # Tighten tolerance to better detect solver discrepancies
    assert np.allclose(X_direct, X_cg, atol=1e-4)

def test_gcv_tuning():
    # Verify that the GCV auto-tuning logic works for alpha and n_frequencies
    X = np.array([
        [1.0, 3.0],
        [np.nan, 6.0],
        [5.0, np.nan],
        [7.0, 9.0],
        [np.nan, 12.0],
        [11.0, np.nan],
        [13.0, 15.0]
    ], dtype=np.float64)
    
    imputer = NufiImputer(method='direct', alpha='auto', n_frequencies='auto', covariance_compensation=True)
    X_filled = imputer.fit_transform(X)
    
    assert not np.any(np.isnan(X_filled))
    assert len(imputer.alphas_) == X.shape[1]
    assert len(imputer.n_frequencies_) == X.shape[1]
    for alpha in imputer.alphas_:
        assert alpha > 0
    for n_freq in imputer.n_frequencies_:
        assert n_freq > 0
        
    # Check imputation quality: values must fall within reasonable bounds (Task 21)
    for col_idx in range(X.shape[1]):
        col_obs = X[:, col_idx]
        obs_min = np.nanmin(col_obs)
        obs_max = np.nanmax(col_obs)
        # Verify no wild outliers
        assert np.all(X_filled[:, col_idx] >= obs_min - 5.0)
        assert np.all(X_filled[:, col_idx] <= obs_max + 5.0)

def test_stochastic_imputation():
    # Verify that stochastic multiple imputation produces non-deterministic filled values
    # on missing spots while preserving non-nan values.
    X = np.array([
        [1.0],
        [np.nan],
        [3.0],
        [np.nan],
        [5.0]
    ], dtype=np.float64)
    
    imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False)
    imputer.fit(X)
    
    X_filled_1 = imputer.transform(X, stochastic=True, stochastic_scale=1.5)
    X_filled_2 = imputer.transform(X, stochastic=True, stochastic_scale=1.5)
    
    assert not np.any(np.isnan(X_filled_1))
    assert not np.any(np.isnan(X_filled_2))
    
    # Preserves original non-nan values
    assert X_filled_1[0, 0] == 1.0
    assert X_filled_2[0, 0] == 1.0
    assert X_filled_1[2, 0] == 3.0
    assert X_filled_2[2, 0] == 3.0
    
    # Missing spots should have different stochastic values
    assert X_filled_1[1, 0] != X_filled_2[1, 0]
    assert X_filled_1[3, 0] != X_filled_2[3, 0]
    
    # With fixed random_state: calls should be reproducible
    imputer_seeded = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=42)
    imputer_seeded.fit(X)
    X_seeded_1 = imputer_seeded.transform(X, stochastic=True, stochastic_scale=1.5)
    X_seeded_2 = imputer_seeded.transform(X, stochastic=True, stochastic_scale=1.5)
    assert np.array_equal(X_seeded_1, X_seeded_2)

def test_stochastic_imputation_multicol():
    # Verify multi-column stochastic imputation maintains valid cross-column relationships (Task 19)
    X = np.array([
        [1.0, 2.0],
        [np.nan, np.nan],
        [3.0, 6.0],
        [np.nan, np.nan],
        [5.0, 10.0]
    ], dtype=np.float64)
    
    imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=True, random_state=42)
    imputer.fit(X)
    X_filled = imputer.transform(X, stochastic=True)
    
    assert not np.any(np.isnan(X_filled))
    # Values should be reasonable
    assert np.all(X_filled[:, 0] >= 0.0)
    assert np.all(X_filled[:, 1] >= 0.0)
    # Cross-column: ratio between columns should be approximately preserved
    obs_mask = ~np.isnan(X).any(axis=1)
    if obs_mask.sum() >= 2:
        observed_ratio = X[obs_mask, 0] / X[obs_mask, 1]
        filled_ratio = X_filled[~obs_mask, 0] / X_filled[~obs_mask, 1]
        assert np.allclose(np.mean(filled_ratio), np.mean(observed_ratio), rtol=0.5)

def test_imputer_edge_cases():
    # Task 22: Missing edge cases
    
    # 1. All-NaN column
    X_all_nan = np.array([
        [1.0, np.nan],
        [2.0, np.nan],
        [3.0, np.nan]
    ], dtype=np.float64)
    imputer1 = NufiImputer(covariance_compensation=True)
    X_filled = imputer1.fit_transform(X_all_nan)
    assert np.isnan(X_filled[:, 1]).all()  # column with all NaNs remains NaN or handles gracefully
    
    # 2. Single-row input
    X_single_row = np.array([[1.0, np.nan]], dtype=np.float64)
    imputer2 = NufiImputer(covariance_compensation=True)
    X_filled_row = imputer2.fit_transform(X_single_row)
    assert np.isnan(X_filled_row[0, 1])  # single row cannot be interpolated but returns gracefully
    
    # 3. Single-column input with all values observed (no NaNs — no-op path)
    X_no_nans = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
    imputer3 = NufiImputer(covariance_compensation=True)
    X_filled_no_nans = imputer3.fit_transform(X_no_nans)
    assert np.allclose(X_no_nans, X_filled_no_nans)
    
    # 4. Invalid parameters: negative/zero alpha should raise ValueError
    with pytest.raises(ValueError):
        bad_imputer = NufiImputer(alpha=-1.0)
        bad_imputer.fit_transform(X_no_nans)
