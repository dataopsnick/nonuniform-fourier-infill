# TESTPLAN.md - Validation, Tests & Device Benchmarking

This test plan defines the formal testing suite, mathematical verification criteria, and hardware-specific benchmark procedures for the `nufi` (Non-Uniform Fourier Infilling) library. It ensures the library's high-performance, GPU-accelerated Fourier-basis spectral reconstruction is both robust and theoretically rigorous.

---

## 1. Objectives & Goals
The validation and testing processes are designed to:
- **Ensure 100% Passing Tests**: Maintain full functional correctness across all components on all supported hardware.
- **Verify Mathematical Correctness**:
  - Prove derivative continuity ($C^\infty$ continuous-time smooth curves).
  - Prove covariance preservation through LDLᵀ decomposition.
  - Verify Generalized Cross-Validation (GCV) auto-tuning selecting optimal bounds.
- **Enforce Agent-Native Quality**:
  - Validate transaction logging, error propagation on logging failure, and snapshot reversion.
- **Benchmark Hardware & Solvers**:
  - Compare CPU vs. GPU (MPS/CUDA) scaling.
  - Compare Iterative Conjugate Gradient (CG) vs. Direct solvers.

---

## 2. Test Suite & Coverage Matrix

### 2.1 Unit Tests (pytest-based)

| Test Module | Component | Verified Functions / Properties |
| :--- | :--- | :--- |
| `tests/test_imputer.py` | `NufiImputer` | `fit`, `transform`, `fit_transform`, Tikhonov L2 solvers, CG convergence, SVD-based GCV optimization, Stochastic posterior process sampling. |
| `tests/test_covariance.py` | Core Math & Covariance | LDLᵀ covariance compensation, spectral reconstruction scale alignment, derivative continuity. |
| `tests/test_agent.py` | Agent-Native Layer | `impute_dataframe` zero-config, JSON diagnostics (SNR, spectral entropy), `plot_diagnostics` visualization, `TransformationTracker` logging, `TransformationLoggingError` on failure, snapshot reversion. |

### 2.2 Integration & Pipeline Tests
- **DataFrame End-to-End Pipeline**: Load dataframe with NaNs -> Run `impute_dataframe` -> Extract diagnostics -> Revert version -> Verify state.
- **MultiIndex Pipeline**: Group-by grouping, chronological sorting, independent group infilling, and multi-index schema restoration.

---

## 3. Mathematical Verification Criteria

### 3.1 Derivative Continuity ($C^\infty$)
Trigonometric basis functions are infinitely differentiable:
$$y(t) = \sum_{k=0}^{M-1} F_k e^{2\pi i f_k t}$$
- **Verification Method**: Using numeric differentiation (`np.diff`), we calculate the first and second derivatives of the reconstructed signal:
  - $\dot{y}(t) = \frac{dy}{dt}$
  - $\ddot{y}(t) = \frac{d^2y}{dt^2}$
- **Pass Criterion**: The resulting derivative arrays must show no sudden jumps or discontinuities (variance of the derivative differences remains bounded and smooth).

### 3.2 Covariance Discrepancy
LDLᵀ decomposition compensates for frequency-domain correlation distortions:
- **Verification Method**: Compare the Frobenius norm error of the infilled covariance matrix $\Sigma_{\text{infilled}}$ vs. the true correlated ground truth $\Sigma_{\text{true}}$:
  $$E_{\text{cov}} = \lVert \Sigma_{\text{true}} - \Sigma_{\text{infilled}} \rVert_F$$
- **Pass Criterion**: $E_{\text{cov}}$ under `covariance_compensation=True` must be significantly lower than standard linear/spline interpolations.

---

## 4. Device & Solver Benchmarking

### 4.1 CPU vs. GPU Scaling
We evaluate performance scaling across device backends:
- **Devices**: CPU, MPS (Apple Silicon), CUDA (NVIDIA GPUs).
- **Metric**: Execution time as a function of the number of samples ($N$).
- **Script**: Located in `tests/benchmark.py` running empirical comparisons.

### 4.2 Iterative CG vs. Direct Solvers
- **Direct Solver**: Evaluates dense matrix inverses via SVD trace, optimal for $N < 2000$.
- **Conjugate Gradient (CG) Solver**: Iteratively solves the Hermitian positive-definite system $(A^H A + \alpha I) F = A^H y$ with $O(N M)$ complexity without explicit matrix storage, scaling to $N \geq 10000$.

---

## 5. Automation & CI/CD Execution
To run the complete test suite locally:
```bash
# Run all unit tests
PYTHONPATH=. pytest

# Run benchmark script to output comparison tables
python tests/benchmark.py
```
