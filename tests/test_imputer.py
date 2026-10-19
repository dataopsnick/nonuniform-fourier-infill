import numpy as np
import pandas as pd
import pytest
from nufi.impute import NufiImputer, pool_rubins_rules
from nufi.wrappers import infill_dataframe, infill_multiindex_dataframe, infill_rbmi_dataframe

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
    assert 'timestamp' not in df_filled.columns
    # Verify observed values are preserved.
    # NOTE: infill_dataframe promotes time_col to the DataFrame index, so rows
    # must be addressed positionally with .iloc (labels are timestamps now).
    assert df_filled['signal'].iloc[0] == 10.0
    assert df_filled['signal'].iloc[2] == 30.0
    # Sanity: filled values should be within a reasonable range of observed data
    assert 5.0 <= df_filled['signal'].iloc[1] <= 35.0, (
        f"Filled value {df_filled['signal'].iloc[1]} is far outside observed range [10, 30]"
    )
    assert 5.0 <= df_filled['signal'].iloc[3] <= 35.0
    
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


def test_multiindex_wrapper_uneven_groups():
    # Groups with different sizes and non-aligned timestamps
    arrays = [
        ['group_A', 'group_A', 'group_A', 'group_B', 'group_B'],
        [1.0, 3.0, 5.0, 2.0, 4.0]
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=('entity', 'time'))
    df = pd.DataFrame({
        'signal': [1.0, np.nan, 5.0, np.nan, 4.0]
    }, index=index)
    
    imputer = NufiImputer(method='direct', covariance_compensation=False)
    df_filled = infill_multiindex_dataframe(df, imputer)
    assert isinstance(df_filled, pd.DataFrame)
    assert not df_filled.isna().any().any()
    # Values in different groups should be imputed independently
    assert df_filled.loc[('group_A', 1.0), 'signal'] == 1.0
    assert df_filled.loc[('group_B', 4.0), 'signal'] == 4.0

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
    # Use a slightly relaxed tolerance to avoid platform‑specific CG convergence variance
    assert np.allclose(X_direct, X_cg, atol=1e-3)

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
        # Verify no wild outliers using a tolerance proportional to the data scale
        col_range = obs_max - obs_min if obs_max > obs_min else 1.0
        # Tighten bounds to catch obviously poor imputations
        assert np.all(X_filled[:, col_idx] >= obs_min - 0.1 * col_range)
        assert np.all(X_filled[:, col_idx] <= obs_max + 0.1 * col_range)
        # Additional sanity: imputed values should not all be identical to the mean
        nan_rows = np.isnan(col_obs)
        if nan_rows.sum() > 1:
            assert np.std(X_filled[nan_rows, col_idx]) > 1e-8, (
                f"Imputed values in column {col_idx} appear constant"
            )

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
    
    # Use seeded imputers with different random states for a deterministic divergence check
    imputer_a = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=42)
    imputer_b = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=99)
    imputer_a.fit(X)
    imputer_b.fit(X)
    X_a = imputer_a.transform(X, stochastic=True, stochastic_scale=1.5)
    X_b = imputer_b.transform(X, stochastic=True, stochastic_scale=1.5)
    assert not np.array_equal(X_a, X_b), "Different random seeds should produce different stochastic imputations"
    
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
        assert np.allclose(np.mean(filled_ratio), np.mean(observed_ratio), rtol=0.3)

def test_imputer_edge_cases():
    # Task 22: Missing edge cases
    
    # 1. All-NaN column
    X_all_nan = np.array([
        [1.0, np.nan],
        [2.0, np.nan],
        [3.0, np.nan]
    ], dtype=np.float64)
    imputer1 = NufiImputer(covariance_compensation=True)
    with pytest.warns(UserWarning, match=r"all.NaN|column.*empty|no valid (observations|values|samples)"):
        X_filled = imputer1.fit_transform(X_all_nan)
    assert np.isnan(X_filled[:, 1]).all()  # column with all NaNs remains NaN or handles gracefully
    assert np.allclose(X_filled[:, 0], X_all_nan[:, 0])  # non-NaN column should be unchanged
    
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

    with pytest.raises(ValueError):
        bad_imputer = NufiImputer(alpha=0.0)
        bad_imputer.fit_transform(X_no_nans)


def _pmm_series():
    t = np.arange(12, dtype=np.float64)
    y = np.sin(0.7 * t) + 0.3 * t
    X = y.reshape(-1, 1).copy()
    X[3, 0] = np.nan
    X[8:, 0] = np.nan
    return t, X


def test_pmm_shift_observed_invariance_and_offset():
    t, X = _pmm_series()
    imputer = NufiImputer(
        method="direct", alpha=1e-4, covariance_compensation=False,
        random_state=42, mnar_strategy="pmm_shift", delta_adjustment=2.5,
    )
    imputer.fit(X)
    base = imputer.transform(X, stochastic=False, delta_adjustment=0.0)
    shifted = imputer.transform(X, stochastic=False)
    nan_mask = np.isnan(X[:, 0])
    assert np.array_equal(base[~nan_mask], shifted[~nan_mask])
    np.testing.assert_allclose(shifted[nan_mask, 0] - base[nan_mask, 0], 2.5, rtol=1e-12)


def test_pmm_slope_and_spectral_dampening():
    t, X = _pmm_series()
    nan_mask = np.isnan(X[:, 0])
    imputer = NufiImputer(
        method="direct", alpha=1e-4, covariance_compensation=False,
        random_state=42, mnar_strategy="pmm_slope", delta_adjustment=0.0,
    )
    imputer.fit(X)
    zero = imputer.transform(X, stochastic=False)
    slope = imputer.transform(X, stochastic=False, delta_adjustment=0.5)
    diff = slope[nan_mask, 0] - zero[nan_mask, 0]
    expected = 0.5 * np.maximum(0.0, t[nan_mask] - 7.0)
    expected[np.where(nan_mask)[0] == 3] = 0.5 * (3.0 - 2.0)
    np.testing.assert_allclose(diff, expected, rtol=1e-12)

    imp_damp = NufiImputer(
        method="direct", alpha=1e-4, covariance_compensation=False,
        random_state=42, mnar_strategy="spectral_dampening", delta_adjustment=0.0,
    )
    imp_damp.fit(X)
    f0 = imp_damp.transform(X, stochastic=False, delta_adjustment=0.0)
    F0 = imp_damp.coefficients_[0].copy()
    f_k = imp_damp.frequencies_[0]
    f3 = imp_damp.transform(X, stochastic=False, delta_adjustment=3.0)
    F3 = imp_damp.coefficients_[0]
    expected_w = np.exp(-3.0 * (f_k / f_k.max()) ** 2)
    np.testing.assert_allclose(np.abs(F3), np.abs(F0) * expected_w, rtol=1e-9)
    assert np.array_equal(f0[~nan_mask], f3[~nan_mask])


def test_batch_multiple_imputations_and_rubins_rules():
    X = np.array([[1.0], [np.nan], [3.0], [np.nan], [5.0], [np.nan], [7.0]], dtype=np.float64)
    imputer = NufiImputer(method="direct", alpha=1e-4, covariance_compensation=False, random_state=42)
    imputer.fit(X)
    draws = imputer.transform(X, stochastic=True, n_imputations=5)
    assert isinstance(draws, np.ndarray) and draws.shape == (5, 7, 1)
    assert not np.array_equal(draws[0], draws[1])
    np.testing.assert_array_equal(draws, imputer.transform(X, stochastic=True, n_imputations=5))

    q = np.array([1.0, 2.0, 4.0])
    u = np.array([0.5, 0.5, 0.5])
    res = pool_rubins_rules(q, u, n_obs=30)
    M = 3
    q_bar, u_bar, b = q.mean(), u.mean(), np.var(q, ddof=1)
    T = u_bar + (1.0 + 1.0 / M) * b
    lam = (1.0 + 1.0 / M) * b / T
    dfcom = 29.0
    tmp = (1.0 - lam) * (1.0 + dfcom) * dfcom
    dof_br = (M - 1) * tmp / ((dfcom + 3.0) * (M - 1) + (lam**2) * tmp)
    r = (1.0 + 1.0 / M) * b / u_bar
    fmi = (r + 2.0 / (dof_br + 3.0)) / (1.0 + r)
    assert np.isclose(res["pooled_estimate"], q_bar, rtol=1e-6)
    assert np.isclose(res["total_variance"], T, rtol=1e-6)
    assert np.isclose(res["dof"], dof_br, rtol=1e-6)
    assert np.isclose(res["fmi"], fmi, rtol=1e-6)


@pytest.mark.parametrize("method", ["CR", "J2R", "CIR", "LMCF"])
def test_rbmi_dataframe_methods(method):
    rng = np.random.RandomState(7)
    times = np.arange(8, dtype=np.float64)
    rows = []
    for arm, base in (("Placebo", 0.0), ("Active", 1.5)):
        for s in range(4):
            subj = f"{arm}_{s}"
            traj = base + 0.3 * times + 0.1 * np.sin(times) + rng.normal(0, 0.05, len(times))
            last_obs = 4 if arm == "Active" else 7
            for i, t in enumerate(times):
                rows.append({
                    "subject": subj, "arm": arm, "time": t,
                    "outcome": traj[i] if i <= last_obs else np.nan,
                })
    df = pd.DataFrame(rows)
    out = infill_rbmi_dataframe(
        df, arm_col="arm", reference_arm="Placebo",
        subject_col="subject", time_col="time",
        rbmi_method=method, n_imputations=3, stochastic=True, random_state=1,
    )
    assert isinstance(out, list) and len(out) == 3
    obs_mask = df["outcome"].notna()
    for d in out:
        assert not d["outcome"].isna().any()
        np.testing.assert_array_equal(d.loc[obs_mask, "outcome"].to_numpy(), df.loc[obs_mask, "outcome"].to_numpy())
