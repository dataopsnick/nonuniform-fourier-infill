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
</tasklist>
```