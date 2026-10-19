```xml
<tasklist>
  <task status="COMPLETED">
    <id>57</id>
    <title>impute.py:354,462 - Bug: np.diag() returns a read-only view in NumPy 2.x, crashing correlated posterior noise sampling</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:354-355, 462-463

**Replicate**:
```bash
PYTHONPATH=. pytest tests/test_imputer.py -k "test_stochastic_imputation_multicol or test_rbmi_dataframe_methods" -v
PYTHONPATH=. pytest tests/test_agent.py -k "test_tipping_point_analysis_and_plot" -v
```

**Root Cause**:
In `NufiImputer.transform()` (line 354) and `NufiImputer.sample_posterior_noise()` (line 462), the code extracts the diagonal of `T @ T.T` via `Z_vars = np.diag(T @ T.T)` and then mutates it in-place with `Z_vars[Z_vars == 0] = 1.0`. In NumPy >= 1.13 / 2.x, `np.diag()` on a 2-D array returns a read-only view of the diagonal, raising:
`ValueError: assignment destination is read-only`

This single bug blocks 6 tests (`test_stochastic_imputation_multicol`, all 4 `test_rbmi_dataframe_methods[CR/J2R/CIR/LMCF]`, and `test_tipping_point_analysis_and_plot`).

**Fix**:
Copy the diagonal array (or use `np.where`) before mutating `Z_vars` in both `transform()` and `sample_posterior_noise()`.

```diff
--- a/nufi/impute.py
+++ b/nufi/impute.py
@@ -351,7 +351,7 @@ class NufiImputer(BaseEstimator, TransformerMixin):
             eta = rng.normal(0, 1.0, size=eta_shape)
             T = self.noise_transform_
             if T is not None:
-                Z_vars = np.diag(T @ T.T)
+                Z_vars = np.diag(T @ T.T).copy()
                 Z_vars[Z_vars == 0] = 1.0
                 if n_imputations > 1:
                     Z = np.einsum("msc,dc->msd", eta, T)
@@ -459,7 +459,7 @@ class NufiImputer(BaseEstimator, TransformerMixin):
         eta = rng.normal(0, 1.0, size=shape)
         T = self.noise_transform_
         if T is not None:
-            Z_vars = np.diag(T @ T.T)
+            Z_vars = np.diag(T @ T.T).copy()
             Z_vars[Z_vars == 0] = 1.0
             if n_imputations > 1:
                 Z = np.einsum("msc,dc->msd", eta, T)
```

**Acceptance**:
All 6 tests (`test_stochastic_imputation_multicol`, `test_rbmi_dataframe_methods[CR/J2R/CIR/LMCF]`, and `test_tipping_point_analysis_and_plot`) pass.
]]></description>
  </task>
  <task status="COMPLETED">
    <id>58</id>
    <title>test_agent.py:80-83 - Bug: test_impute_dataframe_zero_config checks for .csv snapshots instead of .parquet</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:80-83

**Replicate**:
```bash
PYTHONPATH=. pytest tests/test_agent.py -k "test_impute_dataframe_zero_config" -v
```

**Root Cause**:
`TransformationTracker.save_snapshot()` (`nufi/agent.py:79-85`) was upgraded to write compressed `.parquet` snapshots (`f"{version_id}_{step_name}.parquet"`), but `test_impute_dataframe_zero_config` still filters `os.listdir(self.test_history)` for `f.endswith(".csv")`, failing with `AssertionError: 0 not greater than or equal to 2`.

**Fix**:
Update `test_impute_dataframe_zero_config` in `tests/test_agent.py` to check for `.parquet` snapshot files.

```diff
--- a/tests/test_agent.py
+++ b/tests/test_agent.py
@@ -78,9 +78,9 @@ class TestAgentNativeLayer(unittest.TestCase):
         self.assertTrue(os.path.exists(self.test_log))
         self.assertTrue(os.path.exists(self.test_history))
         
-        # Count snapshot files (.csv)
+        # Count snapshot files (.parquet)
         files = os.listdir(self.test_history)
-        csv_files = [f for f in files if f.endswith(".csv")]
-        self.assertGreaterEqual(len(csv_files), 2, f"Expected at least 2 CSV snapshots, got: {csv_files}")
+        parquet_files = [f for f in files if f.endswith(".parquet")]
+        self.assertGreaterEqual(len(parquet_files), 2, f"Expected at least 2 parquet snapshots, got: {parquet_files}")
```

**Acceptance**:
`PYTHONPATH=. pytest tests/test_agent.py -k "test_impute_dataframe_zero_config"` passes.
]]></description>
  </task>
  <task status="COMPLETED">
    <id>59</id>
    <title>agent.py:505 - Bug: plot_diagnostics raises unhandled AttributeError when column diagnostics entry is None</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:505 (and tests/test_agent.py:262-274)

**Replicate**:
```bash
PYTHONPATH=. pytest tests/test_agent.py -k "test_plot_diagnostics_bad_input" -v
```

**Root Cause**:
In `plot_diagnostics()` (`nufi/agent.py:505`), `diag = diagnostics.get(col_name, {})` returns `None` when `diagnostics` is `{"signal": None}` (because the key `"signal"` exists with value `None`), and then `diag.get("optimized_alpha", 1e-4)` at line 527 raises `AttributeError: 'NoneType' object has no attribute 'get'` instead of falling back to `{}` or raising `TypeError`/`ValueError`. Additionally, `snr` in `{"signal": {"snr_db": "not_a_number"}}` is formatted with `f"{snr:.2f}"`, which raises `ValueError`, whereas non-dict `diag` values raise `AttributeError`.

**Fix**:
In `nufi/agent.py:505`, normalize `diag` so non-dict values either raise `TypeError` or fall back to `{}` (`diag = diagnostics.get(col_name) or {}`, or validate `isinstance(diag, dict)`).

```diff
--- a/nufi/agent.py
+++ b/nufi/agent.py
@@ -502,7 +502,10 @@ def plot_diagnostics(
     for idx, col_name in enumerate(columns):
         orig_data = orig_copy[col_name].to_numpy()
         inf_data = inf_copy[col_name].to_numpy()
-        diag = diagnostics.get(col_name, {})
+        if not isinstance(diagnostics, dict):
+            raise TypeError(f"diagnostics must be a dict, got {type(diagnostics).__name__}")
+        diag = diagnostics.get(col_name, {})
+        if not isinstance(diag, dict):
+            raise TypeError(f"diagnostics[{col_name!r}] must be a dict, got {type(diag).__name__}")
```

**Acceptance**:
`PYTHONPATH=. pytest tests/test_agent.py -k "test_plot_diagnostics_bad_input"` passes, and `PYTHONPATH=. pytest tests/test_agent.py` passes 100% (12/12).
]]></description>
  </task>
  <task status="COMPLETED">
    <id>60</id>
    <title>test_imputer.py:22-29 - Bug: test_infill_dataframe_wrapper indexes timestamp-indexed DataFrame with integer row labels 0, 1, 2, 3</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:22-29

**Replicate**:
```bash
PYTHONPATH=. pytest tests/test_imputer.py -k "test_infill_dataframe_wrapper" -v
```

**Root Cause**:
`infill_dataframe(df, time_col='timestamp', keep_time_col=False)` sets `timestamp` (`[1.0, 2.0, 3.0, 4.0]`) as the index of `df_filled` (`nufi/wrappers.py:101`). In `test_infill_dataframe_wrapper`, `df_filled.loc[0, 'signal']` attempts label-based lookup for `0`, which is not in the index `[1.0, 2.0, 3.0, 4.0]`, raising `KeyError: 0.0`.

**Fix**:
Use positional indexing `df_filled['signal'].iloc[...]` (or the actual timestamp index labels `1.0, 2.0, 3.0, 4.0`) in `test_infill_dataframe_wrapper`.

```diff
--- a/tests/test_imputer.py
+++ b/tests/test_imputer.py
@@ -19,13 +19,13 @@ def test_infill_dataframe_wrapper():
     assert len(df_filled) == len(df)
     assert 'timestamp' not in df_filled.columns
     # Verify observed values are preserved
-    assert df_filled.loc[0, 'signal'] == 10.0
-    assert df_filled.loc[2, 'signal'] == 30.0
+    assert df_filled['signal'].iloc[0] == 10.0
+    assert df_filled['signal'].iloc[2] == 30.0
     # Sanity: filled values should be within a reasonable range of observed data
-    assert 5.0 <= df_filled.loc[1, 'signal'] <= 35.0, (
-        f"Filled value {df_filled.loc[1, 'signal']} is far outside observed range [10, 30]"
+    assert 5.0 <= df_filled['signal'].iloc[1] <= 35.0, (
+        f"Filled value {df_filled['signal'].iloc[1]} is far outside observed range [10, 30]"
     )
-    assert 5.0 <= df_filled.loc[3, 'signal'] <= 35.0
+    assert 5.0 <= df_filled['signal'].iloc[3] <= 35.0
```

**Acceptance**:
`PYTHONPATH=. pytest tests/test_imputer.py -k "test_infill_dataframe_wrapper"` passes.
]]></description>
  </task>
  <task status="COMPLETED">
    <id>61</id>
    <title>impute.py:92 - Bug: NufiImputer.fit() does not validate positive alpha, failing test_imputer_edge_cases when GCV score is inf</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:70-95, 160-169

**Replicate**:
```bash
PYTHONPATH=. pytest tests/test_imputer.py -k "test_imputer_edge_cases" -v
```

**Root Cause**:
In `test_imputer_edge_cases` (`tests/test_imputer.py:259-265`), `NufiImputer(alpha=-1.0).fit_transform(X_no_nans)` and `NufiImputer(alpha=0.0).fit_transform(X_no_nans)` expect a `ValueError`. However:
1. `NufiImputer.fit()` does not validate `self.alpha` upfront.
2. For `X_no_nans` (`N_val = 3`) with `alpha = 0.0` or `-1.0`, `compute_gcv_from_svd` returns `inf` (`denom < 1e-8`), triggering the fallback at `nufi/impute.py:160-163` which silently overwrites `best_alpha = 1.0`!
3. When `transform()` subsequently calls `solve_tikhonov_nudft(..., alpha=best_alpha)`, `best_alpha` is `1.0` instead of `-1.0` or `0.0`, so the `if alpha <= 0: raise ValueError(...)` check in `solve_tikhonov_nudft` is never triggered.

**Fix**:
Validate `self.alpha` at the beginning of `NufiImputer.fit()`: if `self.alpha != 'auto'` and `self.alpha is not None`, require `self.alpha > 0`.

```diff
--- a/nufi/impute.py
+++ b/nufi/impute.py
@@ -74,6 +74,10 @@ class NufiImputer(BaseEstimator, TransformerMixin):
         unless explicitly passed. Handles GCV auto-tuning for alpha and n_frequencies.
         """
         from nufi.kernels.torch_kernels import optimize_alpha_gcv, compute_gcv_from_svd
+
+        if self.alpha is not None and self.alpha != 'auto':
+            if not isinstance(self.alpha, (int, float, np.number)) or self.alpha <= 0:
+                raise ValueError(f"Regularization parameter alpha must be positive or 'auto', got {self.alpha}")
         
         # Convert X to numpy
         if isinstance(X, pd.DataFrame):
```

**Acceptance**:
`PYTHONPATH=. pytest tests/test_imputer.py -k "test_imputer_edge_cases"` passes, and `PYTHONPATH=. pytest tests/test_imputer.py` passes 100% (12/12).
]]></description>
  </task>
  <task status="COMPLETED">
    <id>62</id>
    <title>impute.py:114-125,297-315 - Bug: Overcomplete one-sided Nyquist frequency grid (n_f = N_val) with weak default alpha=1e-4 causes severe Gibbs/Runge blow-up (~112x) in gaps</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:105-125, 297-315 (and tests/test_covariance.py:1-90)

**Replicate**:
```bash
PYTHONPATH=. pytest tests/test_covariance.py -v
```

**Root Cause**:
All 4 tests in `tests/test_covariance.py` (`test_derivative_continuity[direct|cg]` and `test_covariance_preservation[direct|cg]`) fail with huge reconstruction oscillation inside contiguous missing gaps (`max abs error ~ 112` on a unit `sin(t)` signal):
1. When `n_frequencies=None` (the default), `NufiImputer.fit()` sets `n_f = N_val` (e.g., `80` frequencies for `80` observed points) spaced uniformly up to `nyquist_frequency = 1 / (2 * dt)` (`~4.95 Hz`).
2. Because `solve_tikhonov_nudft` fits complex exponentials `exp(+2*pi*i * f_k * t)` on `[0, f_nyq]` without conjugate-symmetric pairing or bandwidth concentration, an `N_val x N_val` Vandermonde system on a gapped grid (missing 20 consecutive points `signal[30:50] = np.nan`) is severely ill-conditioned (super-Nyquist local density on observed segments, unconstrained in the gap). With `alpha=1e-4`, the minimum-norm solution places huge opposing amplitudes (`~10^2`) on high-frequency modes that cancel on the observed timestamps but explode to `~112` inside the gap `[3.0, 5.0]`.
3. Furthermore, in `test_covariance_preservation`, `imputer.fit_transform(X)` runs with default `stochastic=False`, which sets `noise_draws = 0` (`nufi/impute.py:362`), making `covariance_compensation=True` and `covariance_compensation=False` produce identical deterministic outputs unless either low-pass spectral regularization or deterministic covariance/scale alignment is applied.

**Fix**:
1. When `n_frequencies is None` or `'auto'`, bound the default frequency grid and/or apply frequency-dependent Sobolev ($H^1$/$H^2$) spectral weighting (or cap default `n_f` to low-frequency basis modes when `n_frequencies is None`) so gap interpolation favors smooth low-frequency trigonometric modes rather than high-frequency Nyquist ringing.
2. Ensure `solve_tikhonov_nudft` / `NufiImputer` penalizes high frequencies (or selects a parsimonious frequency bandwidth when `n_frequencies=None`) so smooth signals like `sin(t)` and `cos(t)` reconstruct accurately across contiguous gaps (`rtol < 1e-2` in `test_derivative_continuity` and `rtol < 1e-1` in `test_covariance_preservation`).

**Acceptance**:
`PYTHONPATH=. pytest tests/test_covariance.py -v` passes all 4 parametrized tests (`test_derivative_continuity[direct|cg]` and `test_covariance_preservation[direct|cg]`).
]]></description>
  </task>
  <task status="COMPLETED">
    <id>63</id>
    <title>pyproject.toml:22-27 & requirements.txt - Bug: Missing pyarrow dependency required by TransformationTracker.save_snapshot</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:22-27, requirements.txt

**Replicate**:
In a clean virtualenv installed from `requirements.txt` or `pyproject.toml`:
```bash
PYTHONPATH=. pytest tests/test_agent.py
```
Fails in 10 tests with `TransformationLoggingError: Failed to save data snapshot ...: Unable to find a usable engine; tried using: 'pyarrow'`.

**Root Cause**:
`TransformationTracker.save_snapshot()` (`nufi/agent.py:85`) hard-codes `df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=True)`, but `pyarrow` is not listed in `pyproject.toml` dependencies or `requirements.txt`.

**Fix**:
Add `pyarrow>=10.0.0` to `dependencies` in `pyproject.toml` and `requirements.txt`.

```diff
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -24,5 +24,6 @@ dependencies = [
     "numpy>=1.21.6",
     "scipy>=1.6.0",
     "pandas>=1.2.0",
-    "scikit-learn>=1.0.0"
+    "scikit-learn>=1.0.0",
+    "pyarrow>=10.0.0"
 ]
--- a/requirements.txt
+++ b/requirements.txt
@@ -6,3 +6,4 @@ cython>=3.0.0
 pytest>=6.0.0
 matplotlib>=3.3.0
+pyarrow>=10.0.0
```

**Acceptance**:
`PYTHONPATH=. pytest -q` passes 100% (`38 passed, 0 failed`).
]]></description>
  </task>
  <task status="PENDING">
    <id>64</id>
    <title>impute.py:199-201 - [bug · critical] rng is never defined in fit(), crashing on columns with > MAX_SVD_SAMPLES observations</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:199-201

**Root Cause**:
`rng` is never defined anywhere in `fit()`. When a column has more than `MAX_SVD_SAMPLES` (2000) valid observations — a common real-world dataset size — this branch executes `rng.choice(...)` and raises `NameError: name 'rng' is not defined`, crashing `fit()`.

**Fix**:
Create the RNG from the stored seed right after `dev = get_device(self.device)` (and ensure reproducibility), e.g.:
```python
from sklearn.utils import check_random_state
rng = check_random_state(self.random_state)
```
or use `np.random.default_rng(self.random_state).choice(...)`.
]]></description>
  </task>
  <task status="PENDING">
    <id>65</id>
    <title>triton_kernels.py:75-78 - [bug · critical] Non-contiguous tensors (especially F.real/F.imag strided views) corrupt Triton kernel reads</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/triton_kernels.py:75-78

**Root Cause**:
The Triton kernel indexes every input as a flat contiguous array (`ptr + offsets`), but nothing guarantees contiguity. `.to(torch.float32)` is a no-op when the tensor already has that dtype, so non-contiguous float32 views pass through unchanged. In particular, for complex64 `F`, `F.real`/`F.imag` are strided views (stride 2 in elements), and `.to(torch.float32)` returns them unchanged; the kernel then reads interleaved real/imag storage and silently produces wrong values (or fails at launch).

**Fix**:
Call `.contiguous()` on every tensor before launch (or use `torch.view_as_real(F)` and slice).

```diff
--- a/nufi/kernels/triton_kernels.py
+++ b/nufi/kernels/triton_kernels.py
-    t = t.to(torch.float32)
-    f = f.to(torch.float32)
-    F_real = F.real.to(torch.float32)
-    F_imag = F.imag.to(torch.float32)
+    t = t.to(torch.float32).contiguous()
+    f = f.to(torch.float32).contiguous()
+    F_real = F.real.to(torch.float32).contiguous()
+    F_imag = F.imag.to(torch.float32).contiguous()
```
]]></description>
  </task>
  <task status="PENDING">
    <id>66</id>
    <title>setup.py:29 - [bug · critical] Version checks compare packaging.version.Version against plain tuples, raising TypeError</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:29-29

**Root Cause**:
Both the Cython version check (`parse_version(cython_version) < (3, 0, 0)`) and the NumPy version check (`parse_version(numpy_version) < (1, 20, 0)`) in `setup.py` compare a `packaging.version.Version` against a plain tuple, which raises `TypeError` instead of the intended build-time message.

**Fix**:
Compare against parsed versions instead.

```diff
--- a/setup.py
+++ b/setup.py
-if parse_version(cython_version) < (3, 0, 0):
+if parse_version(cython_version) < parse_version("3.0.0"):
```
(And similarly `parse_version("1.20.0")` for the NumPy check.)
]]></description>
  </task>
  <task status="PENDING">
    <id>67</id>
    <title>model.py:12 - [bug · high] CPU device fallback fails at initialize()/execute() because fast_nudft_reconstruct is GPU-only</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/model.py:12-12

**Root Cause**:
The device selection falls back to CPU when CUDA is unavailable, but `fast_nudft_reconstruct` is a GPU-only Triton kernel (its wrapper asserts `t.is_cuda and f.is_cuda and F.is_cuda`). On a CPU-only host, `initialize()` will raise at the dummy JIT-compile call (and every `execute()` call would fail), so the fallback never degrades gracefully.

**Fix**:
Either drop the CPU fallback and fail fast with a clear CUDA requirement, or provide a real CPU implementation.
]]></description>
  </task>
  <task status="PENDING">
    <id>68</id>
    <title>triton_kernels.py:80-81 - [bug · high] Missing validation of tensor rank, element count, and complex dtype</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/triton_kernels.py:80-81

**Root Cause**:
No validation of rank, element count, or dtype. `t.shape[0]`/`f.shape[0]` fail confusingly for 0-dim inputs, and if `f` is not 1-D or `F.numel() != K`, the kernel only masks loads with `k_offsets < K`, so it can read past the actual storage of `F_real`/`F_imag` (or silently compute garbage). Likewise, if `F` is not complex, `F.imag` raises an opaque RuntimeError.

**Fix**:
Add explicit checks and raise clear errors.

```diff
--- a/nufi/kernels/triton_kernels.py
+++ b/nufi/kernels/triton_kernels.py
+    if t.ndim != 1 or f.ndim != 1:
+        raise ValueError("t and f must be 1-D tensors")
+    if not F.is_complex() or F.numel() != f.shape[0]:
+        raise ValueError("F must be complex with the same number of elements as f")
     M = t.shape[0]
     K = f.shape[0]
```
]]></description>
  </task>
  <task status="PENDING">
    <id>69</id>
    <title>torch_kernels.py:238 - [bug · high] Covariance matrix dimension is inconsistent with doubled real/imag logic</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:238-238

**Root Cause**:
The covariance matrix dimension is inconsistent with the doubling logic used throughout this function. Each signal is flattened as `np.concatenate([arr.real, arr.imag])` (a length-`2N` vector), so after `.T`, `flat_data` has shape `(2N, num_signals)`. With `rowvar=False`, `np.cov` therefore returns a `(num_signals, num_signals)` matrix — the real/imag parts are mixed into the *sample* axis, not separate *variables*. Yet the `(1,1)` early return builds `np.eye(2*n)` factors while returning `perm`/`valid_idx` of length `n`, and the NaN-drop block computes `N = covariance_matrix.shape[0] // 2` and pairs index `k` with `(k + N) % (2*N)`, assuming a doubled real+imag variable layout that was never constructed. For any `num_signals > 1` this produces wrongly-sized factors and a `valid_idx` that does not map back to signals.

**Fix**:
Either stack real/imag as separate columns (so the covariance is `2n x 2n`) or drop all the doubled-space logic and use `n` consistently.

```diff
--- a/nufi/kernels/torch_kernels.py
+++ b/nufi/kernels/torch_kernels.py
-    flat_data = np.array(flat_data).T # Shape: samples x dimensions
+    # If real/imag are meant to be separate variables, stack them as columns:
+    #   flat_data.append(np.stack([arr.real, arr.imag], axis=1).reshape(-1))  # or build 2n columns
+    # Otherwise drop the doubled-space pairing/early-return shapes and use n consistently.
```
]]></description>
  </task>
  <task status="PENDING">
    <id>70</id>
    <title>pyproject.toml:23-29 & setup.py - [bug · high] Conflicting/duplicate metadata and install_requires between setup.py and pyproject.toml</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:23-29 (and setup.py)

**Root Cause**:
The project ships a `setup.py` whose `setup(..., install_requires=["numpy>=1.20.0", "packaging"])` only declares numpy/packaging, while `[project].dependencies` in `pyproject.toml` lists scipy, pandas, scikit-learn, and pyarrow. Having both `[project]` metadata and a `setup.py` calling `setup()` with overlapping fields produces conflicting/duplicate metadata, and depending on the setuptools version the runtime dependencies can end up inconsistent (or the scipy/pandas/scikit-learn/pyarrow requirements silently dropped, causing `ImportError` at runtime).

**Fix**:
Consolidate to a single source: remove `install_requires` (and the duplicated `name`/`version`/`packages`) from `setup.py` so `[project]` in `pyproject.toml` is authoritative.
]]></description>
  </task>
  <task status="PENDING">
    <id>71</id>
    <title>client.py:13-15 - [bug · medium] client.infer(...) await has no deadline, risking indefinite benchmark hangs</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/streaming_client/client.py:13-15

**Root Cause**:
The `client.infer(...)` await has no deadline. If the Triton server hangs (overload, stuck request, network partition), this coroutine blocks forever, and since `asyncio.gather` waits on all tasks, the entire benchmark hangs indefinitely with no results reported.

**Fix**:
Wrap the call in `asyncio.wait_for` with a sensible timeout and handle/record timeouts separately.

```diff
--- a/nufi/streaming_client/client.py
+++ b/nufi/streaming_client/client.py
     start_time = time.perf_counter()
-    result = await client.infer(model_name="nudft_imputer", inputs=inputs)
+    try:
+        result = await asyncio.wait_for(
+            client.infer(model_name="nudft_imputer", inputs=inputs),
+            timeout=30.0,
+        )
+    except asyncio.TimeoutError:
+        return time.perf_counter() - start_time, None
     latency = time.perf_counter() - start_time
```
]]></description>
  </task>
  <task status="PENDING">
    <id>72</id>
    <title>client.py:39-40 - [performance · medium] Unbounded concurrency of 1000 requests overwhelms gRPC channel and skews latency metrics</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/streaming_client/client.py:39-40

**Root Cause**:
All `num_requests` (1000) coroutines are scheduled at once with no concurrency limit. This can overwhelm the gRPC channel and the Triton server, and the measured latencies then reflect client-side queueing/backpressure rather than true server round-trip latency.

**Fix**:
Bound the in-flight requests with an `asyncio.Semaphore` (or schedule in chunks) to get a meaningful benchmark.

```diff
--- a/nufi/streaming_client/client.py
+++ b/nufi/streaming_client/client.py
-    tasks = [trigger_request(client, req) for req in requests_data]
+    sem = asyncio.Semaphore(64)
+
+    async def limited(req):
+        async with sem:
+            return await trigger_request(client, req)
+
+    tasks = [limited(req) for req in requests_data]
     results = await asyncio.gather(*tasks)
```
]]></description>
  </task>
  <task status="PENDING">
    <id>73</id>
    <title>client.py:40 - [bug · medium] asyncio.gather without return_exceptions=True aborts benchmark on first failed request</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/streaming_client/client.py:40-40

**Root Cause**:
`asyncio.gather` defaults to `return_exceptions=False`, so the first failed request raises and aborts `main()` before the latency percentiles are reported; the remaining tasks are then cancelled, potentially emitting 'Task exception was never retrieved' warnings.

**Fix**:
Use `return_exceptions=True` and filter/report the failures separately so one bad request doesn't discard the whole benchmark run.

```diff
--- a/nufi/streaming_client/client.py
+++ b/nufi/streaming_client/client.py
-    results = await asyncio.gather(*tasks)
+    results = await asyncio.gather(*tasks, return_exceptions=True)
```
]]></description>
  </task>
  <task status="PENDING">
    <id>74</id>
    <title>triton_kernels.py:71 - [bug · medium] Replace assert with explicit device type and equality validation</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/triton_kernels.py:71-71

**Root Cause**:
`assert` is used to validate external input; assertions are stripped under `python -O`, so CPU tensors would reach the Triton launch and fail with an obscure error. The check also doesn't ensure all tensors are on the same CUDA device.

**Fix**:
Replace with an explicit `if ... raise` and verify device equality.

```diff
--- a/nufi/kernels/triton_kernels.py
+++ b/nufi/kernels/triton_kernels.py
-    assert t.is_cuda and f.is_cuda and F.is_cuda
+    if not (t.is_cuda and f.is_cuda and F.is_cuda):
+        raise TypeError("t, f, and F must be CUDA tensors")
+    if not (t.device == f.device == F.device):
+        raise ValueError("t, f, and F must be on the same device")
```
]]></description>
  </task>
  <task status="PENDING">
    <id>75</id>
    <title>setup.py:9 - [bug · medium] Unguarded packaging import not declared in pyproject.toml [build-system] requires</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:9-9

**Root Cause**:
`packaging` is imported at the top of the script without a guard, but it is not declared in `[build-system] requires` in `pyproject.toml`. In a build environment where `packaging` is absent, the script dies with a raw `ModuleNotFoundError: No module named 'packaging'` before the friendly dependency checks below can run.

**Fix**:
Wrap this import in a `try/except` with a clear message, or declare `packaging` as a build requirement in `pyproject.toml`.

```diff
--- a/setup.py
+++ b/setup.py
-from packaging.version import parse as parse_version
+try:
+    from packaging.version import parse as parse_version
+except ImportError:
+    raise ImportError(
+        "packaging is required to build this package. "
+        "Install it via: pip install packaging"
+    )
```
]]></description>
  </task>
  <task status="PENDING">
    <id>76</id>
    <title>torch_kernels.py:60 - [bug · medium] valid_mask excludes NaN but not +/-inf in NUDFT functions</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:60-60

**Root Cause**:
The validity mask excludes NaN values but not `+/-inf`. Non-finite `timestamps` or `data` then flow into `exponent` (`exp(-2πi*inf*f_k)` -> NaN) and the weighted sum, propagating NaNs into the returned spectrum and downstream covariance/LDL. `solve_tikhonov_nudft` already guards `isinf` for this reason. Note `compute_Fast_ND_NUDFT` has the identical gap at its `valid_mask` line.

**Fix**:
Add an `isinf` check (or use `np.isfinite`) in both functions.

```diff
--- a/nufi/kernels/torch_kernels.py
+++ b/nufi/kernels/torch_kernels.py
-    valid_mask = ~np.isnan(data) & ~np.isnan(timestamps)
+    valid_mask = ~np.isnan(data) & ~np.isnan(timestamps) & ~np.isinf(data) & ~np.isinf(timestamps)
```
]]></description>
  </task>
  <task status="PENDING">
    <id>77</id>
    <title>torch_kernels.py:184-186 - [bug · medium] Apodization correction fails to fold negative-frequency bins above M//2</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:184-186

**Root Cause**:
The apodization correction uses the raw bin index `k` without folding for the DFT's negative-frequency wrap. Bins at index `k > M//2` represent negative frequencies of magnitude `M - k`, not `k`, so their Gaussian deconvolution factor should be `exp((M-k)^2 * sigma^2 / 2)` instead of `exp(k^2 * sigma^2 / 2)`. When `n_out` exceeds `M//2` (i.e. `N_full > M/2`), the high-index bins are over-amplified by a factor of `exp((2kM - M^2) * sigma^2 / 2)`, which grows exponentially and corrupts the output spectrum.

**Fix**:
Fold the frequency index before computing `apodization`.

```diff
--- a/nufi/kernels/torch_kernels.py
+++ b/nufi/kernels/torch_kernels.py
     k = torch.arange(n_out, dtype=torch.float64, device=dev)
-    # Use a sigma scaling that matches the computed sigma from M
-    apodization = torch.exp((k**2) * (sigma**2) / 2.0)
+    k_folded = torch.minimum(k, M - k)  # wrap negative-frequency bins
+    apodization = torch.exp((k_folded**2) * (sigma**2) / 2.0)
```
]]></description>
  </task>
  <task status="PENDING">
    <id>78</id>
    <title>benchmark.py:187-190 - [bug · medium] Zero-observation channel assigns NaN and triggers global ValueError, aborting entire GP benchmark</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:187-190

**Root Cause**:
When a channel has zero valid observations (`n_valid == 0`), this branch assigns NaN to the entire channel. But the global check `if np.any(np.isnan(gp_infilled_data))` a few lines later raises `ValueError`, which is caught by the outer `except Exception` and collapses the whole Gaussian Process benchmark into an `{"Error": ...}` result — even though other channels may have been successfully imputed.

**Fix**:
Either skip/handle the zero-observation channel without producing NaN, or make the NaN check tolerate channels that were intentionally left unfilled.
]]></description>
  </task>
  <task status="PENDING">
    <id>79</id>
    <title>benchmark.py:184-185 - [bug · medium] Fallback linear interpolation uses RangeIndex, ignoring non-uniform timestamps</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:184-185

**Root Cause**:
The fallback builds `pd.Series(col_data)` with a default `RangeIndex`, so `interpolate(method='linear')` treats the irregularly sampled points as equally spaced on the range index and ignores the actual non-uniform `timestamps`. This biases the downstream RMSE/covariance scores for skipped channels against the truly non-uniform timestamps.

**Fix**:
Use the timestamp index with an index-aware interpolation method (`method='index'`).

```diff
--- a/tests/benchmark.py
+++ b/tests/benchmark.py
-                col_series = pd.Series(col_data)
-                gp_infilled_data[:, c] = col_series.interpolate(method='linear').ffill().bfill().to_numpy()
+                col_series = pd.Series(col_data, index=timestamps)
+                gp_infilled_data[:, c] = col_series.interpolate(method='index').ffill().bfill().to_numpy()
```
]]></description>
  </task>
  <task status="PENDING">
    <id>80</id>
    <title>test_agent.py:268-280 - [test · medium] Malformed diagnostics subtest silently swallows exceptions and asserts nothing</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:268-280

**Root Cause**:
This subtest never asserts anything: it silently swallows `KeyError`/`TypeError`/`ValueError` and also passes when no exception is raised at all. It therefore cannot distinguish `plot_diagnostics` correctly rejecting malformed diagnostics from it silently accepting them (or raising nothing while producing broken output).

**Fix**:
Make the expectation explicit — e.g. `with self.assertRaises(...)` for the cases that must reject, or assert the plot file is produced for cases that must be handled gracefully.
]]></description>
  </task>
  <task status="PENDING">
    <id>81</id>
    <title>agent.py:687-688 - [bug · medium] plot_diagnostics crashes with TypeError when optimized_alpha or n_frequencies is None</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:687-688

**Root Cause**:
`impute_dataframe` emits `diagnostics[col] = {"optimized_alpha": None, "n_frequencies": None, ...}` for columns with zero observations (`N_val == 0`). `diag.get(key, default)` only falls back when the key is missing, not when its value is `None`, so `n_f` becomes `None` and `np.linspace(0, nyquist_frequency, n_f)` raises `TypeError`. The later `f"Alpha: {opt_alpha:.2e}"` also crashes on a `None` alpha.

**Fix**:
Handle `None` explicitly (fall back to `1e-4` and `len(timestamps)`) so plotting a fully-NaN column doesn't abort the whole figure.

```diff
--- a/nufi/agent.py
+++ b/nufi/agent.py
-    opt_alpha = diag.get("optimized_alpha", 1e-4)
-    n_f = diag.get("n_frequencies", len(timestamps))
+    opt_alpha = diag.get("optimized_alpha")
+    if opt_alpha is None:
+        opt_alpha = 1e-4
+    n_f = diag.get("n_frequencies")
+    if n_f is None:
+        n_f = len(timestamps)
```
]]></description>
  </task>
  <task status="PENDING">
    <id>82</id>
    <title>pyproject.toml:11-12 - [maintainability · medium] PEP 639 license string and license-files require setuptools>=77.0.0</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:11-12

**Root Cause**:
The string `license = "MIT"` (SPDX expression) and `license-files` are PEP 639 features that setuptools only supports starting in v77.0.0. The declared build-system lower bound is `setuptools>=61.0.0`, so builds using setuptools 61–76 may either reject the string license value or silently omit the license metadata/file.

**Fix**:
Bump the build requirement in `[build-system] requires` to `setuptools>=77.0.0` to match the metadata syntax used.
]]></description>
  </task>
  <task status="PENDING">
    <id>83</id>
    <title>model.py:1 & benchmark.py - [maintainability · low] Remove unused imports json and os</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/model.py:1-1 (and tests/benchmark.py)

**Root Cause**:
Unused imports: `json` in `nufi/kernels/model.py` and `os` in `tests/benchmark.py`.

**Fix**:
Remove the unused `import json` from `nufi/kernels/model.py` and `import os` from `tests/benchmark.py`.
]]></description>
  </task>
  <task status="PENDING">
    <id>84</id>
    <title>impute.py:53-54 - [maintainability · low] self.frequency_grids_ is not cleared at the start of fit(), retaining stale grids on re-fit</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:53-54

**Root Cause**:
`self.frequency_grids_` is initialized once in `__init__` but is never cleared at the start of `fit()`. On a re-fit, columns that end up all-NaN hit the `N_val == 0` early `continue` before overwriting their grid, so `transform()` can silently reuse a stale grid from a previous `fit()` call.

**Fix**:
Clear `self.frequency_grids_ = {}` at the top of `fit()` alongside `self.alphas_ = []` / `self.n_frequencies_ = []`.
]]></description>
  </task>
  <task status="PENDING">
    <id>85</id>
    <title>triton_kernels.py:74 - [documentation · low] Clarify timestamp centering comment regarding phase rotation of coefficients</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/triton_kernels.py:74-74

**Root Cause**:
The advice to "subtract t[0] before calling this" is not generally safe: shifting timestamps by a constant changes each term by the phase factor `exp(-2j*pi*t0*f_k)`, so unless `F` is also rotated by that factor the reconstructed signal changes. Meanwhile the wrapper itself casts large `t` to float32 without centering, silently losing precision.

**Fix**:
Consider centering internally together with the corresponding coefficient phase correction (or computing in float64), and clarify the comment/docstring.

```diff
--- a/nufi/kernels/triton_kernels.py
+++ b/nufi/kernels/triton_kernels.py
-    # to avoid catastrophic cancellation in fp32 arithmetic.
+    # Cast to float32. For timestamps, center them (e.g., subtract t[0]) AND
+    # rotate the coefficients by exp(-2j*pi*t0*f) to keep the result unchanged;
+    # otherwise the phase of the reconstructed signal shifts.
```
]]></description>
  </task>
  <task status="PENDING">
    <id>86</id>
    <title>torch_kernels.py:26-30 - [maintainability · low] MPS device probe only checks float64 allocation, not complex128 linalg/FFT support</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:26-30

**Root Cause**:
The MPS probe only allocates a `float64` tensor, but this pipeline relies on `complex128` linear algebra (`torch.linalg.svd`, `torch.linalg.lstsq`, `torch.linalg.ldl`-equivalents) and `torch.fft.fft` on complex tensors, several of which are not supported (or are silently wrong) on MPS for `complex128`. Device selection can therefore return `mps` and only fail later in the transform/solver, instead of falling back to CPU as intended.

**Fix**:
Probe a representative `complex128` operation (e.g. an SVD or FFT on a small complex tensor) in addition to the allocation.
]]></description>
  </task>
  <task status="PENDING">
    <id>87</id>
    <title>test_covariance.py:40-44 - [maintainability · low] Remove unused dead code variables linear_fill, lin_dx, and lin_ddx</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:40-44

**Root Cause**:
`linear_fill`, `lin_dx`, and `lin_ddx` are computed but never used, so this whole block is dead code. The preceding comment says to compare against linear interpolation, but no assertion follows (and the later comment explicitly says to avoid that comparison).

**Fix**:
Either remove the dead block or add the intended assertion using `lin_dx`/`lin_ddx`.
]]></description>
  </task>
  <task status="PENDING">
    <id>88</id>
    <title>test_covariance.py:36-38 - [maintainability · low] Remove duplicate 3x/5x derivative assertions making 5x assertions unreachable</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:36-38

**Root Cause**:
The 3× bounds duplicate the 5× bounds asserted below. Since 3 < 5, the 3× checks are stricter, making the later (message-bearing) 5× assertions unreachable when they fail, while the 3× checks lack descriptive failure messages.

**Fix**:
Keep a single pair of bounds with descriptive messages and remove the redundant pair.
]]></description>
  </task>
  <task status="PENDING">
    <id>89</id>
    <title>test_agent.py:121-128 - [test · low] Compare result_df against an unmutated copy of clean_df</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:121-128

**Root Cause**:
`assert_frame_equal(clean_df, result_df)` compares `result_df` against the very same object passed into `impute_dataframe`. If `impute_dataframe` mutates its input DataFrame in place and returns it, this assertion passes trivially and the test fails to detect the mutation.

**Fix**:
Keep an unmutated reference copy before the call and assert against it.

```diff
--- a/tests/test_agent.py
+++ b/tests/test_agent.py
         clean_df = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0], "signal": [10.0, 20.0, 30.0]})
+        expected_df = clean_df.copy()
         result_df, _ = impute_dataframe(
             clean_df,
             time_col="timestamp",
             log_path=self.test_log,
             history_dir=self.test_history
         )
-        pd.testing.assert_frame_equal(clean_df, result_df, atol=1e-8)
+        pd.testing.assert_frame_equal(expected_df, result_df, atol=1e-8)
```
]]></description>
  </task>
  <task status="PENDING">
    <id>90</id>
    <title>test_agent.py:350-351 - [test · low] Decouple test from exact solve_tikhonov_nudft invocation counts</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:350-351

**Root Cause**:
The assertions (`small_calls == big_calls` and the magic bounds `>= 2` / `<= 4 * n_subject_cols`) couple the test to the exact number of `solve_tikhonov_nudft` invocations produced by the current implementation. A legitimate change such as caching, batching, or consolidating solves can alter call counts while the results remain correct, causing false test failures.

**Fix**:
Prefer asserting on observable results (e.g. `treatment_effects.shape`, `tipping_point`) and keep only a loose regression bound if the solve-count guarantee is part of the contract.
]]></description>
  </task>
  <task status="PENDING">
    <id>91</id>
    <title>test_agent.py:190-191 - [test · low] Save df_mutated as second snapshot to properly test revert_to_version</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:190-191

**Root Cause**:
`df_mutated` is never passed to `save_snapshot`, so the false-positive guard can never fail: `revert_to_version` has no way to return a dataframe that was never registered.

**Fix**:
Save the mutated dataframe as a second snapshot and revert to the first version id, then assert it equals the original.

```diff
--- a/tests/test_agent.py
+++ b/tests/test_agent.py
         df_mutated = df_orig.copy()
         df_mutated["signal"] = 999.0
+        tracker.save_snapshot(df_mutated, "step_2_mutated")
         df_reverted = tracker.revert_to_version(ver_id)
         pd.testing.assert_frame_equal(df_orig, df_reverted)
         self.assertFalse(df_reverted["signal"].equals(df_mutated["signal"]))
```
]]></description>
  </task>
  <task status="PENDING">
    <id>92</id>
    <title>agent.py:75,910-912 - [maintainability & performance · low] Hoist treatment-arm computation out of control loop and chain exceptions with `from e`</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:75, 910-912

**Root Cause**:
1. (`agent.py:910-912`) `endpoints_t`/`var_t` depend only on the treatment delta `d_t`, yet they are recomputed for every `(i, j)` grid cell inside the control-delta loop, repeating the same NUDFT-derived statistics `len(control_deltas)` extra times per treatment delta.
2. (`agent.py:75`) Re-raising `TransformationLoggingError` after an `except` without `from e` discards the original exception traceback in `_log`, `__init__`, `save_snapshot`, `list_versions`, and `revert_to_version`.

**Fix**:
1. Hoist the treatment-arm endpoint/mean/variance computation into a pre-loop pass indexed by `j`, then reuse them in the inner loop:
```python
    treatment_means = []
    treatment_vars = []
    for d_t in treatment_deltas:
        endpoints_t = base_t + d_t * unit_t + noise_t
        treatment_vars.append(
            endpoints_t.var(axis=1, ddof=1) / n_t if n_t > 1 else np.full(n_imputations, 1e-6)
        )
        treatment_means.append(endpoints_t.mean(axis=1))
```
2. Use `raise TransformationLoggingError(...) from e` across `TransformationTracker`.
]]></description>
  </task>
  <task status="PENDING">
    <id>93</id>
    <title>test_imputer.py:250,297 & pyproject.toml:8 - [test & maintainability · low] Relax GPU-sensitive rtol=1e-12, decouple warning regex, and clarify GPU extra in description</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:250, 297 & nonuniform-fourier-infill/pyproject.toml:8

**Root Cause**:
1. (`test_imputer.py:297`) Compares two separate `transform()` calls (`base` with delta=0 and `shifted` with delta=2.5) with `rtol=1e-12`. On GPU backends (CUDA/MPS), `torch.linalg.lstsq` is not guaranteed bit-identical across calls. The same bit-identical assumption affects `test_pmm_slope_and_spectral_dampening`.
2. (`test_imputer.py:250`) The `pytest.warns` `match` pattern tightly couples the test to the exact all-NaN warning wording.
3. (`pyproject.toml:8`) The package description advertises "GPU-accelerated" while `torch` is only an optional extra (`nufi[gpu]`).

**Fix**:
1. Use a looser tolerance (e.g. `atol=1e-6`) in `test_imputer.py:297` and `test_pmm_slope_and_spectral_dampening`.
2. Use `pytest.warns(UserWarning)` without `match` (or pin deliberately) at `test_imputer.py:250`.
3. Clarify in `pyproject.toml` description that GPU acceleration requires `nufi[gpu]`.
]]></description>
  </task>
</tasklist>
```