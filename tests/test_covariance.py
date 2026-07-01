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
    
    # Infill using our imputer
    imputer = NufiImputer(method='direct', covariance_compensation=False)
    infilled = imputer.fit_transform(signal.reshape(-1, 1), timestamps=t).ravel()
    
    # Calculate first and second derivatives numerically
    dx = np.diff(infilled)
    ddx = np.diff(dx)
    
    # Task 23: Tighten derivative thresholds to more realistic bounds
    assert np.max(np.abs(dx)) < 0.2   # tight bound: ~2× the expected max (0.101)
    assert np.max(np.abs(ddx)) < 0.02  # tight bound: ~2× the expected max (0.0102)

    # Task 24: Baseline comparison with linear interpolation
    from scipy.interpolate import interp1d
    valid = ~np.isnan(signal)
    linear_fill = interp1d(t[valid], signal[valid], kind='linear', fill_value='extrapolate')(t)
    lin_dx = np.diff(linear_fill)
    lin_ddx = np.diff(lin_dx)
    
    # Fourier must be smoother (have smaller or equal max derivative spikes) than linear
    assert np.max(np.abs(dx)) <= np.max(np.abs(lin_dx)) * 1.01
    assert np.max(np.abs(ddx)) < np.max(np.abs(lin_ddx))

def test_covariance_preservation():
    # Verify that multi-signal covariance is maintained after imputation
    np.random.seed(42)
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
    
    filled_cov = np.cov(X_filled[:, 0], X_filled[:, 1])
    
    # Task 23: Tighten covariance preservation thresholds to rtol=1e-2, atol=1e-2
    np.testing.assert_allclose(filled_cov, original_cov, rtol=1e-2, atol=1e-2)
