import numpy as np
import pytest
from nufi.impute import NufiImputer

def test_derivative_continuity():
    # Linear interpolation has discontinuous first-derivatives (sudden jumps).
    # Fourier interpolation has smooth derivatives because sine/cosine are infinitely differentiable.
    # We will verify that the first and second numerical derivatives are continuous.
    t = np.linspace(0, 10, 100)
    # Sinusoidal signal with missing segment (NaNs)
    signal = np.sin(t)
    signal[30:50] = np.nan
    
    # TODO: add parametrized tests for boundary NaNs, multiple gaps, and extreme missing ratios
    # Infill using our imputer
    imputer = NufiImputer(method='direct', covariance_compensation=False)
    infilled = imputer.fit_transform(signal.reshape(-1, 1), timestamps=t).ravel()
    assert not np.any(np.isnan(infilled)), "Imputer left NaNs in the output"
    
    # Calculate first and second derivatives numerically
    dx = np.diff(infilled)
    ddx = np.diff(dx)
    
    # Expected max |dx| for sin(t) sampled at dt≈0.101: amplitude × ω × dt ≈ 1 × 1 × 0.101 = 0.101
    # Allow 2× margin for infill artifacts near the gap boundary.
    assert np.max(np.abs(dx)) < 0.2
    # Expected max |ddx| ~ amplitude × ω² × dt² ≈ 1 × 1 × 0.0102 = 0.0102; 2× margin.
    assert np.max(np.abs(ddx)) < 0.02

    # Compare against linear interpolation (should have higher derivative spikes)
    valid = ~np.isnan(signal)
    linear_fill = np.interp(t, t[valid], signal[valid])
    lin_dx = np.diff(linear_fill)
    lin_ddx = np.diff(lin_dx)
    
    # Fourier infill should not introduce larger derivative spikes than linear interpolation.
    assert np.max(np.abs(dx)) <= np.max(np.abs(lin_dx))
    assert np.max(np.abs(ddx)) <= np.max(np.abs(lin_ddx))

def test_covariance_preservation():
    # Verify that multi-signal covariance is maintained after imputation
    t = np.linspace(0, 10, 50)
    s1 = np.sin(t)
    s2 = np.cos(t)
    
    original_cov = np.cov(s1, s2)
    
    # Introduce NaNs
    s1_nan = s1.copy()
    s2_nan = s2.copy()
    s1_nan[10:20] = np.nan
    s2_nan[30:40] = np.nan
    
    X = np.stack([s1_nan, s2_nan], axis=1)
    
    imputer = NufiImputer(method='direct', covariance_compensation=True)
    X_filled = imputer.fit_transform(X, timestamps=t)
    assert not np.any(np.isnan(X_filled)), "Imputer left NaNs in the output"
    
    filled_cov = np.cov(X_filled[:, 0], X_filled[:, 1])
    
    # Relaxed tolerances to account for estimation variance with only 50 samples and ~20% missing data.
    np.testing.assert_allclose(filled_cov, original_cov, rtol=1e-1, atol=1e-1)
