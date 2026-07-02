```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>__init__.py:1-1 - This __init__.py is empty and does not export any symbols...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/__init__.py:1-1

This __init__.py is empty and does not export any symbols from the kernels subpackage
(torch_kernels.py, cy_kernels.pyx). The parent nufi/__init__.py establishes a convention of explicit
imports and an __all__ list. Consider whether public kernel functions (e.g., compute_ND_NUDFT,
solve_tikhonov_nudft, etc.) should be re-exported here for a consistent and convenient user-facing
API.
]]></description>
  </task>
  <task status="COMPLETED">
    <id>2</id>
    <title>benchmark.py:158-181 - Bug: UnboundLocalError when GP benchmark is skipped.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:158-181

**Bug: UnboundLocalError when GP benchmark is skipped.** When `n_valid_max > 500`, the `if` branch
sets `results["Gaussian Process"]` to a skip status but does NOT define `gp_infilled_data` or
`gp_time`. Execution then falls through to lines 137–148 where those variables are used, causing an
`UnboundLocalError`. Fix: either place lines 137–148 entirely inside the `else` block, or
return/continue early in the skip path.

              if n_valid_max > 500:
                  results["Gaussian Process"] = {"Status": f"Skipped: too many valid points ({n_valid_max}) for O(n³) GP"}
+                 gp_rmse = float('nan')
+                 gp_cov_err = float('nan')
+                 gp_time = float('nan')
              else:
                  gp_infilled_data = np.zeros_like(df_truth.to_numpy())
                  for c in range(n_channels):
                      col_data = df_masked.to_numpy()[:, c]
                      valid = ~np.isnan(col_data)
                      n_valid = valid.sum()
                      if n_valid < 2:
-                         # Not enough observations to fit a GP; fill with column mean or NaN
                          gp_infilled_data[:, c] = np.nanmean(col_data) if n_valid > 0 else 0.0
                          continue
                      
                      gp = GaussianProcessRegressor(
                          kernel=RBF(length_scale=np.ptp(timestamps) / np.sqrt(len(timestamps))),
                          alpha=0.1,
                          random_state=42,
                          n_restarts_optimizer=3
                      )
                      gp.fit(timestamps[valid].reshape(-1, 1), col_data[valid])
                      gp_infilled_data[:, c] = gp.predict(timestamps.reshape(-1, 1))
                      
                  gp_time = time.time() - start
              gp_rmse = np.sqrt(np.mean((df_truth.to_numpy() - gp_infilled_data) ** 2))
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>benchmark.py:101-102 - Bug: Cubic spline results are always overwritten by linear+ffill/bfill fallback.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:101-102

**Bug: Cubic spline results are always overwritten by linear+ffill/bfill fallback.** The fallback on
this line executes unconditionally — even when cubic spline interpolation succeeds perfectly (no
NaNs). This means the reported "Cubic Spline" metrics always reflect `cubic → linear → ffill/bfill`
instead of pure cubic interpolation. Fix: guard the fallback with `if remaining_nan > 0:`.

-         # Fallback: use linear, then ffill/bfill to ensure no NaN left
+         # Fallback: use linear, then ffill/bfill only when cubic leaves NaN
+         if remaining_nan > 0:
          spline_infilled = spline_infilled.interpolate(method='linear', axis=0).ffill().bfill()
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>benchmark.py:68-70 - Reliability: NUFI constructor outside try/except block.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:68-70

**Reliability: NUFI constructor outside try/except block.** The `NufiImputer(...)` instantiation on
line 54 is not protected by the surrounding try/except. If the constructor raises (e.g., due to
invalid parameters or missing dependencies), the exception propagates uncaught and terminates the
entire benchmark run. Move the constructor inside the try block.

      start = time.time()
-     nufi = NufiImputer(device='cpu', covariance_compensation=True, n_frequencies='auto', alpha='auto', random_state=42)
      try:
+         nufi = NufiImputer(device='cpu', covariance_compensation=True, n_frequencies='auto', alpha='auto', random_state=42)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>5</id>
    <title>torch_kernels.py:145-152 - Bug: No validation that signals in X_list have equal lengths.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:145-152

**Bug: No validation that signals in X_list have equal lengths.**

`flat_data` is built by concatenating `arr.real` and `arr.imag` for each NUDFT result. If X_list
contains signals with different N, the concatenated vectors have different lengths, and
`np.array(flat_data).T` (line 134) will raise a confusing `ValueError` about ragged array creation.
This error surfaces deep inside NumPy rather than with a clear message.

**Suggestion**: Add an early guard at the top of the function to verify that all signals have the
same `len(data)`:

      # Step 2: Flatten & stack the data (preserving phase information)
      # Move to CPU for covariance and LDL^T since scipy/pandas functions are optimized there
+     # Validate equal signal lengths before stacking
+     lens = [len(tensor) for tensor in X_k_result]
+     if len(set(lens)) > 1:
+         raise ValueError(
+             f"All signals must have the same length, got lengths {lens}. "
+             f"Signals of different lengths cannot be covariance-compensated."
+         )
      flat_data = []
      for tensor in X_k_result:
          arr = tensor.cpu().numpy()
          flat_data.append(np.concatenate([arr.real, arr.imag]))
  
      flat_data = np.array(flat_data).T # Shape: samples x dimensions
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>6</id>
    <title>torch_kernels.py:182-183 - Numerical robustness: Fragile regularization when diag_mean is tiny but positive.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:182-183

**Numerical robustness: Fragile regularization when `diag_mean` is tiny but positive.**

When `diag_mean` is a very small positive value (e.g., 1e-20), `eps = max(1e-10 * diag_mean, 1e-15)
= 1e-15`. This tiny perturbation may be insufficient to guarantee positive-definiteness for the LDL
decomposition, causing `scipy.linalg.ldl` to fail or produce a numerically unstable factorization.

**Suggestion**: Use an absolute floor for the regularization, or use `scipy.linalg.eigh` with a
small shift instead:

```python
diag_mean = max(np.mean(np.diag(covariance_matrix)), 0.0)
eps = max(1e-10 * diag_mean, 1e-10)  # absolute floor of 1e-10
```

      diag_mean = np.mean(np.diag(covariance_matrix))
-     eps = max(1e-10 * diag_mean, 1e-15) if diag_mean > 0 else 1e-10
+     eps = max(1e-10 * max(diag_mean, 0.0), 1e-10)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>7</id>
    <title>torch_kernels.py:308-311 - No memory guard in solve_tikhonov_nudft unlike compute_ND_NUDFT.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:308-311

**No memory guard in `solve_tikhonov_nudft` unlike `compute_ND_NUDFT`.**

`compute_ND_NUDFT` has a `MAX_MEM_N = 10_000` guard (line 76), but `solve_tikhonov_nudft` builds the
full complex `A` matrix of shape `(N, M)` at line 264 without any size check. Even for moderate N
and M (e.g., N=8000, M=5000), `A` consumes ~640 MB in `complex128`. This can cause OOM on GPU or CPU
without warning.

**Suggestion**: Add a size check before constructing A, or use matrix-free operations (e.g., the CG
path already avoids materializing A^H A):

```python
MAX_ELEMENTS = 50_000_000  # ~800 MB in complex128
if N * M > MAX_ELEMENTS:
    if solver != 'cg':
        raise ValueError(
            f"Matrix A shape ({N},{M}) is too large ({N*M} elements). "
            f"Use solver='cg' to avoid materializing the full matrix."
        )
```

      # Build Fourier mapping matrix A: shape (N, M)
      # A_nk = exp(2*pi*i * f_k * t_n)
+     N, M = len(t_timestamps), len(t_f_k)
+     MAX_ELEMENTS = 50_000_000  # ~800 MB for complex128
+     if N * M > MAX_ELEMENTS and solver != 'cg':
+         raise ValueError(
+             f"Matrix A shape ({N},{M}) has {N*M} elements; exceeds memory safety limit. "
+             f"Use solver='cg' to avoid materializing the full matrix."
+         )
      exponent = 2.0j * np.pi * t_timestamps.unsqueeze(1) * t_f_k.unsqueeze(0)
      A = torch.exp(exponent)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>torch_kernels.py:48-64 - Nyquist frequency heuristic silently truncates high-frequency content.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:48-64

**Nyquist frequency heuristic silently truncates high-frequency content.**

The Nyquist frequency is estimated from the median positive sampling interval (lines 60–72). For
irregularly or burstily sampled signals, the median interval can be large, producing a low Nyquist
limit that discards legitimate high-frequency information. There is no way for the caller to
override this heuristic.

**Suggestion**: Allow the caller to pass an explicit `nyquist_frequency` or `f_max` parameter:

```python
def compute_ND_NUDFT(X_list, device=None, nyquist_frequency=None):
```

Then use `nyquist_frequency` when provided, falling back to the heuristic only as a default.

+         # Use caller-provided Nyquist if available, otherwise estimate from sampling
+         if nyquist_frequency is None:
          if len(v_timestamps) > 1:
              # Sort to ensure positive diffs
              sort_idx = np.argsort(v_timestamps)
              sorted_ts = v_timestamps[sort_idx]
              p_n = np.diff(sorted_ts)
              p_n = p_n[p_n > 0]  # keep only positive intervals
              if len(p_n) > 0:
                  median_p = np.median(p_n)
                  nyquist_frequency = 0.5 / max(median_p, 1e-12)
              else:
                  import warnings
                  warnings.warn("Cannot estimate Nyquist frequency; all sampling intervals are zero or negative. Defaulting to 1.0.")
                  nyquist_frequency = 1.0
          else:
              import warnings
              warnings.warn("Only one valid sample; cannot estimate Nyquist frequency. Defaulting to 1.0.")
              nyquist_frequency = 1.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>torch_kernels.py:119-120 - np.interp clips out-of-range values silently, introducing edge artifacts.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:119-120

**`np.interp` clips out-of-range values silently, introducing edge artifacts.**

`uniform_grid` is created from `np.linspace(t_min, t_max, N)`, but due to floating-point rounding,
the first/last points can lie slightly outside `[t_min, t_max]`. `np.interp` clips these to the
boundary values of `v_data`, which can introduce flat regions at the edges of the interpolated
signal and cause spectral leakage in the subsequent FFT.

**Suggestion**: Clamp `uniform_grid` explicitly to `[t_min, t_max]`:

```python
uniform_grid = np.clip(uniform_grid, t_min, t_max)
```
Or use `np.linspace(t_min, t_max, N, endpoint=True)` which should already be in bounds for float64,
but the clamp adds safety.

          uniform_grid = np.linspace(t_min, t_max, N)
+         uniform_grid = np.clip(uniform_grid, t_min, t_max)
          # Ensure timestamps are sorted for np.interp (requires monotonic increasing)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>10</id>
    <title>torch_kernels.py:193-193 - valid_idx return from covariance_compensation is fragile for callers.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:193-193

**`valid_idx` return from `covariance_compensation` is fragile for callers.**

When degenerate columns are dropped, `valid_idx` (indices into the covariance matrix) is returned.
The caller must map these indices back to the signal space via `// 2` (as noted in the inline
comment). If the caller misunderstands the N-real-then-N-imag layout, or if the layout changes in
future refactoring, reconstruction will be silently corrupted. The function also returns `lu, d,
perm` from a *reduced* matrix, but the caller may expect a full-size decomposition.

**Suggestion**: Return a more self-documenting structure (e.g., a named tuple or dataclass) with
clear field names, or include helper logic to expand the reduced LDL factors back to full size. At
minimum, the return type should be documented in the docstring — currently the docstring says
nothing about `valid_idx`.

-     return lu, d, perm, valid_idx  # returns covariance-matrix indices (length up to 2*M); caller must map to signal space via // 2
+     return lu, d, perm, valid_idx  # valid_idx: indices into the full (pre-drop) covariance matrix.
+     # Caller must map back to signal space: signal_idx = valid_idx // 2
+     # and to real/imag component: component = valid_idx % 2  (0=real, 1=imag)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>test_agent.py:62-65 - Fragile snapshot filename check</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:62-65

**Fragile snapshot filename check**: The test asserts existence of hardcoded substrings 'pre_infill'
and 'post_infill' in CSV filenames. If the naming convention in
`impute_dataframe`/`TransformationTracker` changes, this test will break even when the snapshot
functionality is working correctly. Consider using a more robust check, such as verifying the count
of CSV files or checking that snapshot metadata (e.g., a manifest file) records the expected
snapshots, rather than relying on exact substring matches in filenames.

+         # More robust: check snapshot count and existence without hardcoding naming patterns
          files = os.listdir(self.test_history)
          csv_files = [f for f in files if f.endswith(".csv")]
-         self.assertTrue(any("pre_infill" in f for f in csv_files))
-         self.assertTrue(any("post_infill" in f for f in csv_files))
+         self.assertGreaterEqual(len(csv_files), 2, f"Expected at least 2 CSV snapshots, got: {csv_files}")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>test_agent.py:84-87 - Incomplete validation of all-NaN diagnostics</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:84-87

**Incomplete validation of all-NaN diagnostics**: The test only asserts that `stability_flags`
contains `'NO_OBSERVATIONS'`, but does not verify that other diagnostic fields (e.g., `snr_db`,
`spectral_entropy`, `optimized_alpha`, `n_frequencies`) are set to safe sentinel values (like
`-inf`, `NaN`, or `0`). If the imputation logic silently returns garbage values in these fields when
there are no observations, this test will not catch it. Consider adding assertions for all expected
diagnostic keys with appropriate sentinel values, or at minimum verifying the keys exist.

          self.assertIn("signal", diagnostics)
-         flags = diagnostics["signal"]["stability_flags"]
+         col_diag = diagnostics["signal"]
+         self.assertIn("stability_flags", col_diag)
+         flags = col_diag["stability_flags"]
          self.assertIsInstance(flags, list)
          self.assertIn("NO_OBSERVATIONS", flags)
+         # Verify other diagnostic fields have safe sentinel values
+         self.assertIn("snr_db", col_diag)
+         self.assertIn("spectral_entropy", col_diag)
+         self.assertIn("optimized_alpha", col_diag)
+         self.assertIn("n_frequencies", col_diag)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>13</id>
    <title>test_agent.py:154-164 - Missing non-interactive matplotlib backend</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:154-164

**Missing non-interactive matplotlib backend**: The `test_agent_plot_diagnostics` method calls
`plot_diagnostics` with `show_plot=False`, but does not configure a non-interactive matplotlib
backend (e.g., `'Agg'`). In headless CI environments without a display server, this can cause the
test to fail with a `TclError` or similar backend error. Add `import matplotlib;
matplotlib.use('Agg')` at the top of the test method (or in `setUpClass`) to ensure the test runs
reliably in all environments.

      def test_agent_plot_diagnostics(self):
+         # Ensure non-interactive backend for headless CI environments
+         import matplotlib
+         matplotlib.use('Agg')
+         
          # Run infilling
          infilled_df, diagnostics = impute_dataframe(
              self.df,
              time_col="timestamp",
              log_path=self.test_log,
              history_dir=self.test_history
          )
          
          # Test plot rendering with show_plot=False to avoid blocking tests
          save_img = os.path.join(self._tmpdir, "test_diagnostics_plot.png")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>14</id>
    <title>test_agent.py:146-152 - Potential false positive in reversion test</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:146-152

**Potential false positive in reversion test**: The test creates a mutated copy `df_mutated` but
never verifies that `df_reverted` is *different* from `df_mutated`. If `save_snapshot` stores a
reference instead of a deep copy and `revert_to_version` returns the same object, the test would
pass even if the reversion logic is broken (i.e., returning the mutated data instead of the
original). Add an assertion that `df_reverted` does not equal `df_mutated` (e.g., `assert not
df_reverted['signal'].equals(df_mutated['signal'])`) to guard against this false positive.

          # Mutate the dataframe
          df_mutated = df_orig.copy()
          df_mutated["signal"] = 999.0
          
          # Verify reversion returns exactly the original data
          df_reverted = tracker.revert_to_version(ver_id)
          pd.testing.assert_frame_equal(df_orig, df_reverted)
+         # Guard against false positive: ensure reverted is not the mutated version
+         self.assertFalse(
+             df_reverted["signal"].equals(df_mutated["signal"]),
+             "revert_to_version returned the mutated dataframe instead of the original snapshot"
+         )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>15</id>
    <title>test_agent.py:98-98 - Loose tolerance may mask unintended modifications</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:98-98

**Loose tolerance may mask unintended modifications**: `test_impute_dataframe_no_nans` uses
`atol=1e-2` with `assert_frame_equal`. Since the input has no NaN values, the function should return
the data unchanged. A tolerance of 0.01 could hide small unintended numeric drift introduced by the
imputation pipeline. Consider using `atol=1e-8` (or omitting `atol`) to ensure the no-NaN path is
truly identity-preserving up to floating-point precision.

-         pd.testing.assert_frame_equal(clean_df, result_df, atol=1e-2)
+         pd.testing.assert_frame_equal(clean_df, result_df, atol=1e-8)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>16</id>
    <title>agent.py:18-40 - Concurrency Bug: Shared log/history corruption under multi-threaded use.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:18-40

**Concurrency Bug: Shared log/history corruption under multi-threaded use.**

Each call to `impute_dataframe` creates its own `TransformationTracker` instance with a private
`_lock`. When multiple threads call `impute_dataframe` concurrently with the same
`log_path`/`history_dir`, each thread's lock only serializes operations within that tracker instance
— not across instances. Concurrent `save_snapshot()` calls from different threads can interleave CSV
writes and log writes, producing corrupted log entries and orphaned snapshots.

**Suggestion:** Use a module-level or class-level lock (e.g., a `threading.Lock` keyed by resolved
log/history paths in a shared dictionary) to serialize all access to the same underlying files.
Alternatively, consider file-level advisory locking (`fcntl.lockf` / `msvcrt.locking`) or move to a
proper single-writer architecture.

- class TransformationTracker:
-     """
-     Manages append-only transformation logging and snapshot-based dataframe version tracking.
-     Saves snapshots under '.nufi_history/' and logs actions to 'nufi_transformations.log'.
-     """
+ # Consider using a module-level lock keyed by resolved paths:
+ _FILE_LOCKS = {}
+ _FILE_LOCKS_LOCK = threading.Lock()
  
-     def __init__(self, log_path: str = "nufi_transformations.log", history_dir: str = ".nufi_history"):
-         safe_root = os.path.realpath(os.getcwd())
-         # Resolve paths and validate immediately:
-         for p in (log_path, history_dir):
-             resolved = os.path.realpath(p)
-             try:
-                 if os.path.commonpath([safe_root, resolved]) != safe_root:
-                     raise ValueError(f"Path {p} is outside the allowed directory.")
-             except ValueError:
-                 raise ValueError(f"Path {p} is outside the allowed directory (possibly on a different drive).")
-         # Store resolved paths (these are canonical and safe for writes)
-         self.log_path = os.path.realpath(log_path)
-         self.history_dir = os.path.realpath(history_dir)
-         self._lock = threading.RLock()
-         # NOTE: This lock is thread-safe only. Concurrent processes sharing the same
-         # log/history paths will corrupt files. Use file locking or dedicated IPC
-         # if cross-process safety is required.
+ class TransformationTracker:
+     def __init__(self, log_path, history_dir):
+         ...
+         with _FILE_LOCKS_LOCK:
+             key = (self.log_path, self.history_dir)
+             if key not in _FILE_LOCKS:
+                 _FILE_LOCKS[key] = threading.RLock()
+             self._lock = _FILE_LOCKS[key]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>17</id>
    <title>agent.py:63-82 - Orphan snapshot on logging failure.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:63-82

**Orphan snapshot on logging failure.**

When `save_snapshot()` successfully writes the CSV file but then
`self.log_transformation(log_entry)` raises a `TransformationLoggingError`, the CSV file remains on
disk with no corresponding log entry. The snapshot becomes untracked — it will never be listed by
`list_versions()` and cannot be reverted to via `revert_to_version()`, yet it consumes disk space
indefinitely.

**Suggestion:** If logging fails, attempt to delete the already-written CSV file before re-raising.
This keeps the history directory consistent: either both the CSV and log entry exist, or neither
does.

          with self._lock:
-             # Write CSV first (if it fails, no log pollution)
              try:
                  df.to_csv(filepath, index=True)
              except Exception as e:
                  raise TransformationLoggingError(f"Failed to save data snapshot {filepath}: {e}")
-             # Log only after successful write to avoid orphan entries
              log_entry = {
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "event": "snapshot_saved",
                  "version_id": version_id,
                  "step_name": step_name,
                  "columns": list(df.columns),
                  "shape": df.shape,
                  "filepath": filepath
              }
              try:
                  self.log_transformation(log_entry)
              except Exception as e:
+                 # Clean up orphaned CSV to keep history consistent
+                 try:
+                     os.remove(filepath)
+                 except OSError:
+                     pass
                  raise TransformationLoggingError(f"Failed to write to transformation log: {e}")
]]></description>
  </task>
  <task status="COMPLETED">
    <id>18</id>
    <title>agent.py:295-314 - Exception masking: original error can be lost during failure logging.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:295-314

**Exception masking: original error can be lost during failure logging.**

In the `impute_dataframe` exception handler, if `tracker.log_transformation(...)` itself raises a
`TransformationLoggingError`, that new exception propagates and replaces the original error from
`imputer.fit()` / `imputer.transform()`. The bare `raise` on the next line is never reached.

**Suggestion:** Wrap the failure-logging attempt in its own try/except so the original exception
always propagates unchanged.

      try:
          imputer.fit(df_copy, timestamps=timestamps)
          infilled_df = imputer.transform(df_copy, timestamps=timestamps, stochastic=stochastic, stochastic_scale=stochastic_scale)
      except Exception:
+         try:
          tracker.log_transformation({
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "event": "infilled_dataframe_failed",
              "pre_infill_version": pre_ver,
              "parameters": {
                  "method": method,
                  "device": str(device),
                  "n_frequencies": n_frequencies,
                  "alpha": alpha,
                  "solver": solver,
                  "covariance_compensation": covariance_compensation,
                  "stochastic": stochastic,
                  "stochastic_scale": stochastic_scale
              }
          })
+         except Exception:
+             pass  # Logging failure is non-fatal; preserve the original error
          raise
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>19</id>
    <title>agent.py:366-383 - Diagnostic inconsistency: missing covariance compensation in fallback PSD computation.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:366-383

**Diagnostic inconsistency: missing covariance compensation in fallback PSD computation.**

When `imputer.reconstructed_` is available, the diagnostics use reconstruction and coefficients that
already include covariance compensation (as noted in the comment on line ~252). However, when
`reconstructed_` is absent (the `else` branch), `solve_tikhonov_nudft` is called directly and no
covariance compensation is applied. This means SNR, spectral entropy, and stability flags may differ
between the compensated-imputation result and the diagnostic fallback path, potentially producing
misleading stability flags (e.g., flagging `POTENTIAL_OVERFIT_LOW_REGULARIZATION` when the actual
compensated signal is fine, or vice versa).

**Suggestion:** Document this discrepancy clearly, or implement covariance compensation in the
fallback branch to match the imputation pipeline. Even better, ensure `imputer.reconstructed_` is
always populated after `fit()` so the fallback is never needed.

          if len(v_data) > 0 and hasattr(imputer, 'reconstructed_') and col_idx in imputer.reconstructed_:
              reconstructed_np = imputer.reconstructed_[col_idx][valid_mask]
              F_np = imputer.coefficients_[col_idx]
          else:
+             # WARNING: fallback PSD does NOT apply covariance compensation.
+             # SNR/entropy/flags may differ from the actual compensated imputation.
              F = solve_tikhonov_nudft(
                  v_timestamps, v_data, f_k, opt_alpha,
                  solver=solver, max_iter=max_iter, tol=tol, device=device
              )
              F_np = F.cpu().numpy()
  
              t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
              t_times = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
              exponent = 2.0j * np.pi * t_times.unsqueeze(1) * t_f_k.unsqueeze(0)
              reconstructed = torch.real(torch.sum(F.unsqueeze(0) * torch.exp(exponent), dim=1))
              reconstructed_np = reconstructed.cpu().numpy()
- 
-         # Note: imputer.reconstructed_ already includes covariance compensation.
-         # Do not apply cov_scale again to avoid double-compensation in diagnostics.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>20</id>
    <title>agent.py:320-329 - Late validation: duplicate time_col check runs after expensive imputation.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:320-329

**Late validation: duplicate `time_col` check runs after expensive imputation.**

The check for duplicate `time_col` values in DataFrame columns (lines ~283-286) occurs AFTER the
imputation has already completed. If this check fails, all the computation time spent on `fit()` and
`transform()` is wasted.

**Suggestion:** Move this validation to the beginning of `impute_dataframe`, right after the initial
`df` type/empty checks.

      # Restore original index/columns name or structure if time_col was used
      if time_col is not None:
          infilled_df = infilled_df.reset_index().rename(columns={'index': time_col})
  
      # Generate JSON diagnostic metadata and column details
      diagnostics = {}
      dev = get_device(device)
-     if time_col is not None and list(df.columns).count(time_col) > 1:
-         raise ValueError(f"time_col '{time_col}' appears multiple times in DataFrame columns. "
-                          f"Ensure column names are unique.")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>21</id>
    <title>test_covariance.py:71-71 - Flaky tolerance risk</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:71-71

**Flaky tolerance risk**: With only 50 samples (~20% missing, i.e., ~10 missing per signal), the
covariance estimates have high variance. The `rtol=5e-2` on diagonal entries (variance ~0.5) means
only ~0.025 absolute tolerance — this can easily exceed estimation noise. If the off-diagonal
(covariance between sin and cos) is near zero, `atol=5e-2` provides some buffer, but the combination
is still fragile. Consider increasing tolerances (e.g., `rtol=1e-1, atol=1e-1`) or using a larger
sample size like 200+ points to stabilize estimates.

**Context**: This matches focus area #1.

-     np.testing.assert_allclose(filled_cov, original_cov, rtol=5e-2, atol=5e-2)
+     np.testing.assert_allclose(filled_cov, original_cov, rtol=1e-1, atol=1e-1)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>22</id>
    <title>test_covariance.py:45-46 - Brittle assertion logic</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:45-46

**Brittle assertion logic**: The assertion `max(|dx|) <= max(|lin_dx|)` assumes Fourier infill never
produces larger first/second difference spikes than linear interpolation. While typically true for
smooth signals like sin(t), this is not mathematically guaranteed — Fourier reconstruction near gap
boundaries can exhibit small overshoots (Gibbs-like ringing), which may produce marginally higher
differences than linear interpolation. A brittle assertion here could cause sporadic CI failures or
mask real correctness issues. Consider changing to a ratio-based check (e.g., Fourier max ≤ 1.5 ×
linear max) to allow a small margin, or verify that both methods stay within reasonable bounds
independently.

**Context**: This matches focus area #4.

-     assert np.max(np.abs(dx)) <= np.max(np.abs(lin_dx))
-     assert np.max(np.abs(ddx)) <= np.max(np.abs(lin_ddx))
+     # Allow small margin: Fourier should be smoother, but not guaranteed to be strictly ≤ linear
+     assert np.max(np.abs(dx)) <= 1.5 * np.max(np.abs(lin_dx))
+     assert np.max(np.abs(ddx)) <= 1.5 * np.max(np.abs(lin_ddx))
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>23</id>
    <title>test_covariance.py:18-18 - Missing method coverage</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:18-18

**Missing method coverage**: Only `method='direct'` is exercised across both tests. Other imputation
methods (e.g., `'cg'`, `'fft'`, etc. if supported by `NufiImputer`) receive zero test coverage and
could silently break. Consider adding `@pytest.mark.parametrize("method", [...])` to run both tests
over all supported methods.

**Context**: This matches focus area #3.

+     # TODO: parameterize over all supported methods, e.g.:
+     # @pytest.mark.parametrize("method", ["direct", "cg", "fft"])
      imputer = NufiImputer(method='direct', covariance_compensation=False)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>24</id>
    <title>test_covariance.py:14-15 - Uncovered edge cases</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:14-15

**Uncovered edge cases**: The TODO here correctly identifies gaps: boundary NaNs (leading/trailing),
multiple disjoint gaps, and extreme missing ratios (>80%) are not tested. These scenarios can expose
real bugs — e.g., `np.interp` used as a baseline on line 37 extrapolates the boundary value for
out-of-range points, which may mask imputer failures at boundaries. Consider adding parametrized
tests for these cases, especially NaN at array edges where many interpolation libraries behave
differently.

**Context**: This matches focus area #2.

      # TODO: add parametrized tests for boundary NaNs, multiple gaps, and extreme missing ratios
      # See: https://github.com/example/issues/123 for tracking
+     # NOTE: `np.interp` baseline on line 37 extrapolates boundary values for out-of-range
+     # points, so boundary-NaN tests should use a different baseline or verify the imputer directly.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>25</id>
    <title>impute.py:148-151 - Bug: best_n_freq = min(max(5, N_val), N_val) always eva...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:148-151

Bug: `best_n_freq = min(max(5, N_val), N_val)` always evaluates to `N_val` for any N_val >= 1.

- If N_val < 5: max(5, N_val) = 5, min(5, N_val) = N_val.
- If N_val >= 5: max(5, N_val) = N_val, min(N_val, N_val) = N_val.

The comment says "avoid underdetermined system" but the expression never caps N_val to a safe value.
This means when all GCV candidates fail SVD (e.g., N_val is very small), the fallback still produces
an n_f that may equal or exceed N_val, risking an underdetermined system that will likely fail again
in `transform()`.

Suggestion: Use `min(N_val // 2, N_val)` or `min(max(5, N_val // 2), N_val - 1)` to ensure n_f <
N_val, or at minimum document that a subsequent solver failure is expected.

              if best_gcv == float('inf'):
                  import warnings
-                 best_n_freq = min(max(5, N_val), N_val)  # avoid underdetermined system
+                 best_n_freq = max(5, min(N_val - 1, N_val // 2))  # ensure n_f < N_val to avoid underdetermined system
                  best_alpha = 1.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>26</id>
    <title>impute.py:183-185 - Potential indexing error: The comment states that valid_...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:183-185

Potential indexing error: The comment states that `valid_idx_comp` references indices in the
"doubled (real+imag) space" from `covariance_compensation()`. If this space has size `2 *
len(X_list)`, then `valid_cols[idx]` will cause an `IndexError` for any `idx >= len(valid_cols)`.

Even if `covariance_compensation` returns indices in the original (non-doubled) space, there is
ambiguity. The code should clarify the contract and ideally add a bounds check or assertion.

Suggestion: Add an assertion like `assert all(0 <= idx < len(valid_cols) for idx in valid_idx_comp),
"valid_idx_comp indices out of valid_cols bounds"` and verify the external function's return
semantics.

                  # Filter valid_cols using valid_idx_comp to handle degenerate columns dropped.
-                 # valid_idx_comp references the doubled (real+imag) space.
-                 actual_valid_cols = [valid_cols[idx] for idx in valid_idx_comp]
+                 # Note: valid_idx_comp should reference indices in the original (non-doubled) space.
+                 # If it references doubled space, this will cause an IndexError.
+                 actual_valid_cols = [valid_cols[idx] for idx in valid_idx_comp if idx < len(valid_cols)]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>27</id>
    <title>impute.py:308-309 - Deterministic path ignores covariance compensation (cov_...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:308-309

Deterministic path ignores covariance compensation (`cov_scale`). When
`self.covariance_compensation=True`, the stochastic branch scales the reconstruction by `cov_scale`,
computes residuals in the compensated space, then unscales. The deterministic (non-stochastic)
branch writes `reconstructed_np` directly, bypassing any diagonal scaling from `self.d_`.

While the net signal value for NaN positions is the same in both paths (scaling cancels out), the
inconsistency means that if `cov_scale` affected the reconstruction beyond simple scaling (e.g., if
the full LDL^T decomposition were applied in the future), the deterministic path would silently
produce different results. The init docstring already acknowledges this partially implemented
feature.

Suggestion: Either apply `cov_scale` consistently in both branches (even if it cancels) for clarity,
or clearly document that covariance compensation only affects the stochastic noise distribution.

                  else:
+                     # Deterministic fill: use reconstructed signal (covariance compensation
+                     # is a no-op for the signal mean; it only affects stochastic noise scaling)
                      X_data[nan_mask, col_idx] = reconstructed_np[nan_mask]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>28</id>
    <title>impute.py:208-215 - Missing check_is_fitted guard. The transform() method...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:208-215

Missing `check_is_fitted` guard. The `transform()` method accesses `self.alphas_`,
`self.n_frequencies_`, and `self.timestamps_` without verifying that `fit()` has been called. If
`transform` is invoked on an unfitted instance, the error will be an unhelpful `AttributeError`
rather than a clear scikit-learn `NotFittedError`.

Suggestion: Add `sklearn.utils.validation.check_is_fitted(self, ['alphas_', 'n_frequencies_',
'timestamps_'])` at the start of `transform()`.

      def transform(self, X, timestamps=None, stochastic=False, stochastic_scale=1.0):
          """
          Transforms X by infilling NaNs using the fitted NUDFT-based smooth reconstruction.
          Supports stochastic posterior sampling representing imputation uncertainty.
          """
+         from sklearn.utils.validation import check_is_fitted
+         check_is_fitted(self, ['alphas_', 'n_frequencies_', 'timestamps_'])
          from nufi.kernels.torch_kernels import solve_tikhonov_nudft
          
          if isinstance(X, pd.DataFrame):
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>29</id>
    <title>impute.py:104-113 - Code duplication: The timestamp sorting and Nyquist frequ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:104-113

Code duplication: The timestamp sorting and Nyquist frequency computation logic (lines ~109-115 in
fit, lines ~236-242 in transform) is duplicated verbatim. If the sorting logic changes (e.g.,
edge-case handling for non-positive diff), both locations must be updated.

Suggestion: Extract this into a private helper `_compute_nyquist_frequency(timestamps, data)` that
returns sorted arrays and the frequency vector.

-             if len(v_timestamps) > 1:
-                 # Ensure sorted before computing sampling intervals
-                 if not np.all(np.diff(v_timestamps) >= 0):
-                     sort_idx = np.argsort(v_timestamps)
-                     v_timestamps = v_timestamps[sort_idx]
-                     v_data = v_data[sort_idx]
-             p_n = np.diff(v_timestamps)
-             min_p = np.min(p_n[p_n > 0]) if np.any(p_n > 0) else 1.0
-             max_sampling_rate = 1.0 / min_p
-             nyquist_frequency = max_sampling_rate / 2.0
+             v_timestamps, v_data = self._sort_and_compute_nyquist(v_timestamps, v_data)
+             nyquist_frequency = v_timestamps[-1]  # placeholder — extract helper
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>30</id>
    <title>wrappers.py:62-76 - Missing NaN validation before setting time_col as index. ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:62-76

Missing NaN validation before setting time_col as index. If `time_col` contains NaN values, Pandas
will create a float64 index with NaN entries. The `NufiImputer` uses the index as timestamps, and
NaN timestamps will cause silent corruption or cryptic downstream errors in frequency estimation.
Add a pre-check like `if pd_df[time_col].isna().any(): raise ValueError(...)` before set_index.

      if time_col is not None:
          if time_col not in pd_df.columns:
              raise ValueError(
                  f"time_col '{time_col}' not found in DataFrame columns: {list(pd_df.columns)}"
+             )
+         if pd_df[time_col].isna().any():
+             raise ValueError(
+                 f"time_col '{time_col}' contains NaN values, which are not valid as timestamps."
              )
          # Capture the original index name before replacement and warn if discarded
          previous_index_name = pd_df.index.name
          if previous_index_name is not None:
              import warnings
              warnings.warn(
                  f"DataFrame index name {previous_index_name!r} is being discarded "
                  "as it is being replaced by time_col.",
                  UserWarning
              )
          if keep_time_col:
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>31</id>
    <title>wrappers.py:76-89 - When keep_time_col=True, the time column is duplicated ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:76-89

When `keep_time_col=True`, the time column is duplicated as both the DataFrame index and a feature
column. After `fit_transform`, if the feature column had missing values, the imputer fills them —
but the index is left unchanged (indices are not imputed). This creates a silent inconsistency: the
index and the column named `time_col` will hold different values. Consider dropping the duplicated
feature column after imputation, or validating that time_col has no missing values before
proceeding.

          if keep_time_col:
              import warnings
              warnings.warn(
                  "keep_time_col=True duplicates timestamps as both index and feature. "
                  "Timestamp magnitudes (e.g., Unix nanoseconds) may dominate covariance "
                  "estimation and produce biased imputations for other columns. "
                  "Consider normalizing timestamps or using keep_time_col=False.",
                  UserWarning
              )
              time_values = pd_df[time_col].copy()
              col_pos = pd_df.columns.get_loc(time_col)
              pd_df = pd_df.set_index(time_col)
              pd_df.index.name = None  # avoid name collision with the column
              pd_df.insert(col_pos, time_col, time_values)
+             # After fit_transform below, the feature column may be imputed while the
+             # index is not. Drop the feature column post-imputation to stay consistent.
+             _drop_time_col_after = True
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>32</id>
    <title>wrappers.py:51-59 - infill_dataframe does not validate that timestamps are ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:51-59

`infill_dataframe` does not validate that timestamps are strictly increasing (monotonic), unlike
`infill_multiindex_dataframe` which raises a clear error for non-positive timestamp diffs.
Non-monotonic or duplicate timestamps will silently produce incorrect frequency estimates and
imputations. Add a monotonicity check analogous to the one in `infill_multiindex_dataframe`.

      if sort:
          if time_col is not None:
              if time_col not in pd_df.columns:
                  raise ValueError(
                      f"time_col '{time_col}' not found in DataFrame columns: {list(pd_df.columns)}"
                  )
              pd_df = pd_df.sort_values(time_col)
          else:
              pd_df = pd_df.sort_index()
+     else:
+         import warnings
+         warnings.warn(
+             "sort=False: timestamps will not be sorted. "
+             "Non-monotonic timestamps may produce incorrect frequency estimates.",
+             UserWarning
+         )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>33</id>
    <title>wrappers.py:195-203 - Casting timestamps to float64 via astype(np.float64) ca...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:195-203

Casting timestamps to float64 via `astype(np.float64)` can lose precision for datetime64[ns] values
far from the Unix epoch (e.g., years beyond ~2242). `np.diff` then computes differences on lossy
conversions, which can produce false zeros or negatives under edge cases, triggering a spurious
monotonicity error. Prefer computing diffs directly on the integer representation:
`np.diff(timestamps.astype('datetime64[ns]').view('int64'))`.

          # Validate strictly monotonic timestamps to prevent Nyquist overflow
          if len(timestamps) > 1:
+             # Use integer diff on datetime64 to avoid float64 precision loss
+             if np.issubdtype(timestamps.dtype, np.datetime64):
+                 diffs = np.diff(timestamps.view('int64'))
+             else:
              diffs = np.diff(timestamps.astype(np.float64))
              if np.any(diffs <= 0):
                  raise ValueError(
                      f"Timestamps for group must be strictly increasing; "
                      f"found non-positive or zero difference. "
                      f"Check for duplicate or out-of-order timestamps."
                  )
]]></description>
  </task>
  <task status="COMPLETED">
    <id>34</id>
    <title>wrappers.py:106-111 - np.array_equal returns False when either index contai...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:106-111

`np.array_equal` returns `False` when either index contains NaN (NaN != NaN by IEEE 754 semantics).
If the DataFrame index has NaN entries — which can happen if `time_col` had NaN and was set as index
— this check will fail with the misleading message "reordered rows". The actual problem is NaN in
the index, not reordering. Consider using `np.array_equal(..., equal_nan=True)` or adding a prior
NaN check on the index.

      # Verify row order before continuing
-     if not np.array_equal(infilled_pd.index, pd_df.index):
+     if not np.array_equal(infilled_pd.index, pd_df.index, equal_nan=True):
          raise ValueError(
              "NufiImputer.fit_transform reordered rows. "
              "Row order must be preserved."
          )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>35</id>
    <title>wrappers.py:177-181 - The sort=False path in infill_multiindex_dataframe si...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:177-181

The `sort=False` path in `infill_multiindex_dataframe` silently accepts unsorted timestamps, which
the docstring acknowledges may produce incorrect results. Consider emitting a warning (consistent
with the suggestion for `infill_dataframe`) to alert callers when they opt out of sorting.

          # We need to sort index by time level to ensure proper chronological order
          if sort:
              group_sorted = group.sort_index(level=time_level)
          else:
+             import warnings
+             warnings.warn(
+                 "sort=False: timestamps will not be sorted. "
+                 "Non-monotonic timestamps may produce incorrect frequency estimates.",
+                 UserWarning
+             )
              group_sorted = group
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>36</id>
    <title>test_imputer.py:134-136 - GCV tuning outlier bounds too lenient (3× col_range).</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:134-136

**GCV tuning outlier bounds too lenient (3× col_range).** Using `3.0 * col_range` as tolerance
allows imputed values to be wildly outside the observed range. For example, with data in [1, 15]
(range=14), values as extreme as -41 or 57 would pass this assertion — providing essentially no
quality guarantee. Consider tightening the bound to something like `0.5 * col_range` or checking
that imputed values fall within `[obs_min - 0.25*col_range, obs_max + 0.25*col_range]` to actually
catch unrealistic imputations.

          col_range = obs_max - obs_min if obs_max > obs_min else 1.0
-         assert np.all(X_filled[:, col_idx] >= obs_min - 3.0 * col_range)
-         assert np.all(X_filled[:, col_idx] <= obs_max + 3.0 * col_range)
+         assert np.all(X_filled[:, col_idx] >= obs_min - 0.5 * col_range)
+         assert np.all(X_filled[:, col_idx] <= obs_max + 0.5 * col_range)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>37</id>
    <title>test_imputer.py:199-199 - rtol=0.5 is too permissive for cross-column ratio validation.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:199-199

**rtol=0.5 is too permissive for cross-column ratio validation.** This allows the mean filled ratio
to differ from the mean observed ratio by up to 50%, which does not meaningfully verify that
cross-column relationships are preserved. A tighter tolerance (e.g., `rtol=0.2` or `rtol=0.3`) would
provide a more useful regression guard without being overly strict for stochastic draws.

-         assert np.allclose(np.mean(filled_ratio), np.mean(observed_ratio), rtol=0.5)
+         assert np.allclose(np.mean(filled_ratio), np.mean(observed_ratio), rtol=0.3)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>38</id>
    <title>test_imputer.py:210-213 - All-NaN column test does not verify correctness of non-NaN columns.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:210-213

**All-NaN column test does not verify correctness of non-NaN columns.** The test only asserts that
the all-NaN column (column 1) remains NaN, but it does not check that column 0 — which has valid
observed values and no NaNs — is returned unchanged. If a bug caused the imputer to corrupt adjacent
columns when encountering an all-NaN column, this test would not catch it. Add an assertion
verifying column 0 is unchanged.

      imputer1 = NufiImputer(covariance_compensation=True)
      with pytest.warns(UserWarning, match="all-NaN|empty|no valid"):
          X_filled = imputer1.fit_transform(X_all_nan)
      assert np.isnan(X_filled[:, 1]).all()  # column with all NaNs remains NaN or handles gracefully
+     assert np.allclose(X_filled[:, 0], X_all_nan[:, 0])  # non-NaN column should be unchanged
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>39</id>
    <title>test_imputer.py:227-230 - Invalid-parameter test only covers negative alpha; zero...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:227-230

**Invalid-parameter test only covers negative alpha; zero alpha is also invalid and more
dangerous.** `alpha=0` (non-positive but not negative) could trigger division-by-zero in the
Tikhonov regularization path inside the solver without being caught by a `ValueError`. Add an
explicit test for `alpha=0` to ensure the imputer rejects it.

      # 4. Invalid parameters: negative/zero alpha should raise ValueError
      with pytest.raises(ValueError):
          bad_imputer = NufiImputer(alpha=-1.0)
+         bad_imputer.fit_transform(X_no_nans)
+ 
+     with pytest.raises(ValueError):
+         bad_imputer = NufiImputer(alpha=0.0)
          bad_imputer.fit_transform(X_no_nans)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>40</id>
    <title>test_imputer.py:165-167 - Stochastic test uses or when checking that two missing positions differ across runs.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:165-167

**Stochastic test uses `or` when checking that two missing positions differ across runs.** If only
one of `X_filled_1[1,0]` vs `X_filled_2[1,0]` differs while `X_filled_1[3,0]` equals
`X_filled_2[3,0]`, the assertion still passes, providing a weaker guarantee than intended. Consider
checking that *both* positions differ, or at minimum that the vectors are not identical overall.

-     assert abs(X_filled_1[1, 0] - X_filled_2[1, 0]) > 1e-12 or abs(X_filled_1[3, 0] - X_filled_2[3, 0]) > 1e-12, (
-         "Stochastic imputations should differ; both pairs were identical"
+     assert abs(X_filled_1[1, 0] - X_filled_2[1, 0]) > 1e-12 and abs(X_filled_1[3, 0] - X_filled_2[3, 0]) > 1e-12, (
+         "Stochastic imputations should differ at both missing positions; at least one pair was identical"
      )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>41</id>
    <title>test_imputer.py:16-19 - keep_time_col=False path is not verified.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:16-19

**`keep_time_col=False` path is not verified.** The test checks that `keep_time_col=True` retains
the `timestamp` column, but it never asserts that `keep_time_col=False` (the default tested earlier)
actually removes `timestamp` from the output. If a regression caused the column to be retained in
both modes, this test would still pass. Add an explicit assertion.

      df_filled = infill_dataframe(df, time_col='timestamp', keep_time_col=False)
      assert isinstance(df_filled, pd.DataFrame)
      assert not df_filled.isna().any().any()
      assert len(df_filled) == len(df)
+     assert 'timestamp' not in df_filled.columns
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>42</id>
    <title>setup.py:7-7 - Build-time import may fail</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:7-7

**Build-time import may fail**: `from packaging.version import Version` requires the `packaging`
package at build time, but it is not declared in `pyproject.toml`'s `[build-system] requires`. When
building from source (e.g., `pip install .`), the import will fail with `ImportError` if `packaging`
is not already installed in the build environment. Add `"packaging"` to the build-system requires in
`pyproject.toml`, or alternatively perform version comparisons without relying on `packaging` (e.g.,
`tuple(map(int, version.split('.')))`).

+ # Consider using a packaging-free version check, or ensure
+ # packaging is listed in pyproject.toml [build-system] requires:
+ #   requires = [..., "packaging"]
  from packaging.version import Version
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>43</id>
    <title>setup.py:91-93 - Redundant import warnings</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:91-93

**Redundant `import warnings`**: `warnings` is imported individually on lines 90, 115, 118, and 131
deep inside conditional blocks. This clutters the code and risks missing imports if the control flow
changes. Import `warnings` once at the top of the file alongside the other standard-library imports.

+ # At the top of the file:
              import warnings
+ 
+ # Then later, simply use:
              warnings.warn(
                  "OpenMP not found. Install libomp via 'brew install libomp' "
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>44</id>
    <title>setup.py:126-131 - Linux OpenMP detection swallows compiler diagnostics</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:126-131

**Linux OpenMP detection swallows compiler diagnostics**: When the test compilation fails,
`res.stderr` is never surfaced to the user — only a generic warning is emitted. This makes it hard
to troubleshoot *why* OpenMP is unavailable (e.g., missing `libgomp-dev` vs. an unrelated compiler
error). Consider including `res.stderr` in the warning message or raising a more specific warning.

          if has_openmp:
              ext_compiler_args = ["-fopenmp"]
              ext_linker_args = ["-fopenmp"]
          else:
-             import warnings
-             warnings.warn("OpenMP not supported by compiler, disabling.")
+             warnings.warn(
+                 f"OpenMP test compilation failed (returncode={res.returncode}). "
+                 f"Compiler stderr: {res.stderr.decode().strip()}. OpenMP disabled."
+             )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>45</id>
    <title>setup.py:76-81 - macOS libomp detection only checks for the header file</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:76-81

**macOS libomp detection only checks for the header file** (`include/omp.h`) but does not verify
that the corresponding library (e.g., `libomp.dylib`) exists in `lib/`. If the header is present but
the library is missing — for example after an incomplete Homebrew installation — the build will fail
at link time with a confusing error. Consider also checking for
`os.path.exists(os.path.join(candidate, "lib", "libomp.dylib"))` (or `libomp.so`) before accepting a
candidate.

          for candidate in libomp_candidates:
              if candidate and os.path.isdir(os.path.join(candidate, "include")):
                  # Ensure we have omp.h to avoid matching sys.prefix false positives
                  if os.path.exists(os.path.join(candidate, "include", "omp.h")):
+                     lib_dir = os.path.join(candidate, "lib")
+                     lib_name = "libomp.dylib" if sys.platform == "darwin" else "libomp.so"
+                     if os.path.exists(os.path.join(lib_dir, lib_name)):
                      libomp_path = candidate
                      break
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>46</id>
    <title>pyproject.toml:2-2 - Unnecessary build dependency: Cython is unused.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:2-2

**Unnecessary build dependency: Cython is unused.** No `.pyx`, `.pxd` files or `cythonize()` calls
were found anywhere in the project. Requiring `Cython>=3.0.0` adds compilation time for every `pip
install`, risks build failures on platforms where Cython is not easily available (e.g., some minimal
containers), and misleads readers into thinking C extension modules exist.

**Suggestion:** Remove `"Cython>=3.0.0"` from `build-system.requires` unless C extension modules are
added.

- requires = ["setuptools>=61.0.0", "wheel", "Cython>=3.0.0", "oldest-supported-numpy"]
+ requires = ["setuptools>=61.0.0", "wheel"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>47</id>
    <title>pyproject.toml:2-2 - oldest-supported-numpy is likely unnecessary without C extensions.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:2-2

**`oldest-supported-numpy` is likely unnecessary without C extensions.** This meta-package exists to
build C extensions against the oldest NumPy ABI for binary compatibility. If no Cython or C
extensions link against NumPy, this dependency is dead weight and adds a misleading build
requirement.

**Suggestion:** Remove `"oldest-supported-numpy"` along with `"Cython>=3.0.0"` (see above).

- requires = ["setuptools>=61.0.0", "wheel", "Cython>=3.0.0", "oldest-supported-numpy"]
+ requires = ["setuptools>=61.0.0", "wheel"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>48</id>
    <title>pyproject.toml:22-28 - torch as a mandatory dependency forces a 2+ GB install on all users.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:22-28

**`torch` as a mandatory dependency forces a 2+ GB install on all users.** The project description
itself states GPU acceleration is optional and requires a separate PyTorch install. For CPU-only
users, this is a massive, unnecessary dependency. It also ties the package to a specific torch
version range globally.

**Suggestion:** Move `torch>=2.0.0` to `[project.optional-dependencies]` under a meaningful extras
group (e.g., `gpu` or `all`), and note in the docs that GPU users must install `nufi[gpu]` or `pip
install torch` separately.

  dependencies = [
      "numpy>=1.21.6",
      "scipy>=1.6.0",
      "pandas>=1.2.0",
      "scikit-learn>=1.0.0",
-     "torch>=2.0.0"
  ]
+ 
+ [project.optional-dependencies]
+ gpu = ["torch>=2.0.0"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>49</id>
    <title>pyproject.toml:30-34 - Empty gpu extras group does nothing.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:30-34

**Empty `gpu` extras group does nothing.** `pip install nufi[gpu]` installs no additional packages.
The comment says "intentionally empty — torch manages its own CUDA runtime", but this is confusing:
users expect `[gpu]` to enable GPU support, and the empty list silently does nothing, potentially
leaving users with a non-GPU build of torch or none at all.

**Suggestion:** Either move `torch` here (see above) so `pip install nufi[gpu]` actually installs
PyTorch, or remove the `gpu` extras group entirely to avoid misleading users.

  [project.optional-dependencies]
- # GPU acceleration requires a CUDA-capable or MPS-capable PyTorch wheel.
- # Install separately, e.g.: pip install torch --index-url https://download.pytorch.org/whl/cu118
- # This extras group is intentionally empty — torch manages its own CUDA runtime.
- gpu = []
+ # GPU acceleration; installs PyTorch. For CUDA/MPS-specific wheels, override:
+ #   pip install nufi[gpu] --index-url https://download.pytorch.org/whl/cu118
+ gpu = ["torch>=2.0.0"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>50</id>
    <title>HEAD:0-0 - Personal/infrastructure data exposure: The log entry cont...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/.git/logs/refs/remotes/origin/HEAD:0-0

Personal/infrastructure data exposure: The log entry contains a user identity with a hostname that
appears to be a Kubernetes pod name (r-dataopsnick-ocr-chat-ui-interface-uznotwfj-f29fc-5pds5). This
file resides in .git/logs/ and should never be committed to a repository. If exposed, it leaks
internal infrastructure naming conventions, username, and hostname patterns.
]]></description>
  </task>
</tasklist>

```