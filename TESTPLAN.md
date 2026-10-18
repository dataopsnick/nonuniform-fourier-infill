# TESTPLAN.md - Validation, Tests & Device Benchmarking

This test plan defines the formal testing suite, mathematical verification criteria, and hardware-specific benchmark procedures for the `nufi` (Non-Uniform Fourier Infilling) library. It ensures the library's high-performance, GPU-accelerated Fourier-basis spectral reconstruction and clinical trial MNAR sensitivity tools are both robust and theoretically rigorous.

---

## 1. Objectives & Goals
The validation and testing processes are designed to:
- **Ensure 100% Passing Tests**: Maintain full functional correctness across all components on all supported hardware.
- **Verify Mathematical Correctness**:
  - Prove derivative continuity ($C^\infty$ continuous-time smooth curves).
  - Prove covariance preservation through LDLᵀ decomposition.
  - Verify Generalized Cross-Validation (GCV) auto-tuning selecting optimal bounds.
  - Verify Clinical Trial MNAR Sensitivity Models (Pattern Mixture Models, Reference-Based Multiple Imputation, Tipping Point Sensitivity Analysis, and Rubin's Rules pooling).
- **Enforce Agent-Native Quality**:
  - Validate transaction logging, error propagation on logging failure, Diggle-Kenward MNAR diagnostic flags, and snapshot reversion.
- **Benchmark Hardware & Solvers**:
  - Compare CPU vs. GPU (MPS/CUDA) scaling.
  - Compare Iterative Conjugate Gradient (CG) vs. Direct solvers.
  - Compare NUFI PMM/RBMI against standard baselines under MCAR, MAR, and MNAR clinical dropout mechanisms.

---

## 2. Test Suite & Coverage Matrix

### 2.1 Unit Tests (pytest-based)

| Test Module | Component | Verified Functions / Properties |
| :--- | :--- | :--- |
| `tests/test_imputer.py` | `NufiImputer` & Wrappers | `fit`, `transform`, `fit_transform`, Tikhonov L2 solvers, CG convergence, SVD-based GCV optimization, Stochastic posterior process sampling, Pattern Mixture Models (`pmm_shift`, `pmm_slope`, `delta_adjustment`), Reference-Based Multiple Imputation (`infill_rbmi_dataframe`: `CR`, `J2R`, `CIR`, `LMCF`), Batch MI (`n_imputations=M`), and `pool_rubins_rules`. |
| `tests/test_covariance.py` | Core Math & Covariance | LDLᵀ covariance compensation, spectral reconstruction scale alignment, derivative continuity ($C^\infty$). |
| `tests/test_agent.py` | Agent-Native Layer | `impute_dataframe` zero-config, JSON diagnostics (SNR, spectral entropy, Diggle-Kenward `MNAR_DROPOUT_LEVEL_BIAS` & `MNAR_DROPOUT_SLOPE_BIAS` flags), `run_tipping_point_analysis`, `plot_diagnostics` & `plot_tipping_point` visualizations, `TransformationTracker` logging, `TransformationLoggingError` on failure, snapshot reversion. |

### 2.2 Integration & Pipeline Tests
- **DataFrame End-to-End Pipeline**: Load dataframe with NaNs -> Run `impute_dataframe` -> Extract diagnostics -> Revert version -> Verify state.
- **MultiIndex Pipeline**: Group-by grouping, chronological sorting, independent group infilling, and multi-index schema restoration.
- **Clinical Trial Estimand / MNAR Pipeline**: Load multi-arm longitudinal trial panel with monotone dropouts -> Run `infill_rbmi_dataframe(rbmi_method='CIR', n_imputations=50)` -> Pool treatment effect via `pool_rubins_rules` -> Run `run_tipping_point_analysis` over a 2D $\delta$-grid -> Verify tipping point threshold $\delta^*$ and audit log lineage.

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

### 3.3 Pattern Mixture Models (PMM) & Reference-Based Multiple Imputation (RBMI)
Under MNAR clinical trial discontinuation at dropout time $t_d$:
- **PMM Shift & Slope**:
  $$x_{\text{PMM}}(t) = \tilde{x}(t) + \delta_{\text{shift}} + \delta_{\text{slope}} \cdot \max(0, t - t_d) + Z_{\text{scaled}}(t), \quad t > t_d$$
- **Copy Increments in Reference (CIR)**:
  $$x_{\text{CIR}}(t) = x_{\text{active}}(t_d) + \int_{t_d}^{t} \dot{x}_{\text{ref}}(\tau)\,d\tau + Z_{\text{ref}}(t) = x_{\text{active}}(t_d) + \left(\tilde{x}_{\text{ref}}(t) - \tilde{x}_{\text{ref}}(t_d)\right) + Z_{\text{ref}}(t), \quad t > t_d$$
- **Pass Criteria**:
  1. **Observed Invariance**: Observed data points ($t \le t_d$) remain strictly unchanged (`np.array_equal`).
  2. **Boundary Continuity ($C^0$) & Smoothness ($C^\infty$)**: For `CIR` and `pmm_slope`, right-hand limit $\lim_{t \to t_d^+} \mathbb{E}[x(t)] = x(t_d)$, and post-dropout increments match the analytical Fourier derivative of the reference arm within `atol=1e-6`.
  3. **Jump to Reference (`J2R`)**: Post-dropout mean trajectory converges to $\tilde{x}_{\text{ref}}(t)$ for $t > t_d$.

### 3.4 Rubin's Rules Pooling & Tipping Point Sensitivity Analysis
Given $M \ge 2$ stochastic posterior imputations yielding point estimates $\hat{Q}_1, \dots, \hat{Q}_M$ and within-imputation variances $U_1, \dots, U_M$:
$$\bar{Q} = \frac{1}{M}\sum_{m=1}^{M} \hat{Q}_m, \quad \bar{U} = \frac{1}{M}\sum_{m=1}^{M} U_m, \quad B = \frac{1}{M-1}\sum_{m=1}^{M} (\hat{Q}_m - \bar{Q})^2, \quad T = \bar{U} + \left(1 + \frac{1}{M}\right)B$$
- **Pass Criteria**:
  1. `pool_rubins_rules` matches closed-form values for $\bar{Q}$, $T$, Fraction of Missing Information ($\text{FMI}$), and Barnard-Rubin degrees of freedom within `rtol=1e-6`.
  2. `run_tipping_point_analysis` monotonically shifts the pooled $t$-statistic as $\delta$ increases against the treatment arm, accurately identifying the tipping point boundary $\delta^*$ where $p \ge \alpha_{\text{level}}$ with zero redundant calls to `solve_tikhonov_nudft`.

---

## 4. Device & Solver Benchmarking

### 4.1 CPU vs. GPU Scaling
We evaluate performance scaling across device backends:
- **Devices**: CPU, MPS (Apple Silicon), CUDA (NVIDIA GPUs).
- **Metric**: Execution time as a function of the number of samples ($N$) and number of sensitivity grid points ($|\Delta| \times M$).
- **Script**: Located in `tests/benchmark.py` running empirical comparisons.

### 4.2 Iterative CG vs. Direct Solvers
- **Direct Solver**: Evaluates dense matrix inverses via SVD trace, optimal for $N < 2000$.
- **Conjugate Gradient (CG) Solver**: Iteratively solves the Hermitian positive-definite system $(A^H A + \alpha I) F = A^H y$ with $O(N M)$ complexity without explicit matrix storage, scaling to $N \geq 10000$.

### 4.3 MNAR Clinical Dropout Benchmark
- Evaluates treatment effect estimation bias ($|\hat{\theta} - \theta_{\text{true}}|
$), RMSE, and covariance error when post-baseline dropout is triggered by steep unfavorable trajectories (MNAR), comparing `nufi` (`pmm_slope` / `RBMI-CIR`) against naive MAR imputation (Cubic Spline, MICE, Gaussian Processes).

---

## 5. Automation & CI/CD Execution
To run the complete test suite locally:
```bash
# Run all unit tests
PYTHONPATH=. pytest

# Run benchmark script to output comparison tables
python tests/benchmark.py