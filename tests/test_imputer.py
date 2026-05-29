import numpy as np
import pandas as pd
import pytest
from nufi.impute import NufiImputer
from nufi.wrappers import infill_dataframe, infill_multiindex_dataframe

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
    assert len(imputer.alphas_) == 2
    assert len(imputer.n_frequencies_) == 2
    for alpha in imputer.alphas_:
        assert alpha > 0
    for n_freq in imputer.n_frequencies_:
        assert n_freq > 0

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
