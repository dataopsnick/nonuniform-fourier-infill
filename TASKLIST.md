```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>pyproject.toml:20-21 - The gpu extra only lists torch&gt;=2.0.0, which is alrea...</title>
    <description><![CDATA[
### Location: pyproject.toml:20-21

The `gpu` extra only lists `torch>=2.0.0`, which is already a core dependency (line 17). This makes
the optional dependency a no-op. If the intent is to allow torch-free installations, `torch` should
be removed from `dependencies` and placed only here. Otherwise, this section should be removed as
redundant.

  [project.optional-dependencies]
  gpu = ["torch>=2.0.0"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>benchmark.py:98-98 - Since pandas&gt;=1.2.0 is required, the ffill() and bfi...</title>
    <description><![CDATA[
### Location: tests/benchmark.py:98-98

Since `pandas>=1.2.0` is required, the `ffill()` and `bfill()` methods are available and are the
preferred (non-deprecated) API. Using `fillna(method='ffill')` triggers a `FutureWarning` in pandas
1.4+ and will be removed in future versions. The original `.ffill().bfill()` was already correct.

-         spline_infilled = spline_infilled.interpolate(method='linear', axis=0).fillna(method='ffill').fillna(method='bfill')
+         spline_infilled = spline_infilled.interpolate(method='linear', axis=0).ffill().bfill()
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>test_covariance.py:35-35 - The strict inequality np.max(np.abs(ddx)) &lt; np.max(np.ab...</title>
    <description><![CDATA[
### Location: tests/test_covariance.py:35-35

The strict inequality `np.max(np.abs(ddx)) < np.max(np.abs(lin_ddx))` has no floating-point
tolerance. If both are numerically very close (or identical due to edge cases), this assertion
becomes fragile. Consider adding a small relative/absolute tolerance (e.g., `np.max(np.abs(ddx)) <
np.max(np.abs(lin_ddx)) * 0.99`) or using `<=` with an epsilon to make the test robust against
floating-point variations.

-     assert np.max(np.abs(ddx)) < np.max(np.abs(lin_ddx))
+     assert np.max(np.abs(ddx)) < np.max(np.abs(lin_ddx)) * 0.99
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>setup.py:34-36 - The check os.environ.get("NUIFI_NO_OPENMP") uses Python...</title>
    <description><![CDATA[
### Location: setup.py:34-36

The check `os.environ.get("NUIFI_NO_OPENMP")` uses Python truthiness, so an empty string (`""`) or
`"0"` will be treated as falsy and OpenMP will remain enabled — contrary to user intent. Since the
variable is named `NUIFI_NO_OPENMP`, its mere presence should be sufficient to indicate disabling.
Use `"NUIFI_NO_OPENMP" in os.environ` to check for key existence instead, which handles empty
strings and other edge cases correctly.

- if os.environ.get("NUIFI_NO_OPENMP"):
+ if "NUIFI_NO_OPENMP" in os.environ:
      ext_compiler_args = []
      ext_linker_args = []
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>5</id>
    <title>setup.py:57-57 - Using print() for build-time warnings bypasses Python's...</title>
    <description><![CDATA[
### Location: setup.py:57-57

Using `print()` for build-time warnings bypasses Python's standard warning filtering and writes to
stdout rather than stderr. Consider using `import warnings` and `warnings.warn(...)` instead, which
allows users to suppress or escalate these warnings via standard Python warning filters (e.g., `-W`
flag or `PYTHONWARNINGS`).

-             print("WARNING: OpenMP not found. Install libomp via 'brew install libomp' for better performance.")
+             import warnings
+             warnings.warn("OpenMP not found. Install libomp via 'brew install libomp' for better performance.")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>6</id>
    <title>setup.py:86-86 - Using print() for build-time warnings bypasses Python's...</title>
    <description><![CDATA[
### Location: setup.py:86-86

Using `print()` for build-time warnings bypasses Python's standard warning filtering and writes to
stdout rather than stderr. Consider using `warnings.warn(...)` instead, which allows users to
suppress or escalate these warnings via standard Python warning filters.

-             print("WARNING: OpenMP not supported by compiler, disabling.")
+             warnings.warn("OpenMP not supported by compiler, disabling.")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>7</id>
    <title>setup.py:43-46 - Only two Homebrew paths are checked for libomp. Users who...</title>
    <description><![CDATA[
### Location: setup.py:43-46

Only two Homebrew paths are checked for libomp. Users who installed libomp via MacPorts
(`/opt/local/lib/libomp`) or a custom prefix will not get OpenMP support despite having it
available. Consider also checking `os.environ.get("LIBOMP_PATH")` or running `brew --prefix libomp`
as a subprocess to auto-detect the correct path.

          libomp_candidates = [
              "/opt/homebrew/opt/libomp",   # Apple Silicon Homebrew
              "/usr/local/opt/libomp",       # Intel Homebrew
+             "/opt/local/lib/libomp",       # MacPorts
          ]
+         # Also allow explicit override via environment variable
+         env_libomp = os.environ.get("LIBOMP_PATH")
+         if env_libomp:
+             libomp_candidates.insert(0, env_libomp)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>wrappers.py:40-43 - When keep_time_col=True, set_index(time_col) makes th...</title>
    <description><![CDATA[
### Location: nufi/wrappers.py:40-43

When `keep_time_col=True`, `set_index(time_col)` makes the index inherit the column name (e.g.,
`'timestamp'`), and then `pd_df[time_col] = time_values` creates a column with the same name. This
results in the index and a column sharing an identical label, which is ambiguous and violates pandas
conventions. Operations like `df.reset_index()` would fail with a conflict, and `df['timestamp']` is
ambiguous.

Consider giving the index a distinct name to avoid collision, e.g., set `pd_df.index.name = None` or
rename it to something like `f'_{time_col}_index'` after the `set_index` call.

          if keep_time_col:
              time_values = pd_df[time_col].copy()
              pd_df = pd_df.set_index(time_col)
+             pd_df.index.name = None  # avoid name collision with the column
              pd_df[time_col] = time_values
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>agent.py:78-88 - Backward-incompatible parsing: old-format snapshots (with...</title>
    <description><![CDATA[
### Location: nufi/agent.py:78-88

Backward-incompatible parsing: old-format snapshots (without UUIDs) whose step names contain
underscores are parsed incorrectly.

Old format: `ver_{timestamp}_{step_name}.csv`
New format: `ver_{timestamp}_{uuid_hex}_{step_name}.csv`

When an old-format file like `ver_1680000000_pre_infill.csv` (step_name="pre_infill") is parsed,
`parts` = ['ver','1680000000','pre','infill.csv'] (len=4), so it matches the `len(parts) >= 4`
branch. This incorrectly treats 'pre' as a UUID segment, producing version_id='ver_1680000000_pre'
instead of 'ver_1680000000'. This makes the snapshot unreachable via `revert_to_version()` and
silently breaks access to existing history.

Suggestion: detect whether parts[2] is an 8-char hex UUID (e.g., `re.match(r'^[0-9a-f]{8}$',
parts[2])`). If not, fall back to the old 2-segment version_id logic.

                  for f in files:
                      parts = f.split("_")
-                     if len(parts) >= 4:
+                     # New format: ver_{timestamp}_{uuid_hex8}_{step_name}.csv
+                     # Old format: ver_{timestamp}_{step_name}.csv
+                     if len(parts) >= 4 and re.match(r'^[0-9a-f]{8}$', parts[2]):
                          version_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                          step_name = "_".join(parts[3:]).replace(".csv", "")
-                     elif len(parts) == 3:
+                     elif len(parts) >= 3:
                          version_id = f"{parts[0]}_{parts[1]}"
-                         step_name = parts[2].replace(".csv", "")
+                         step_name = "_".join(parts[2:]).replace(".csv", "")
                      else:
                          version_id = parts[0]
                          step_name = parts[1].replace(".csv", "")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>10</id>
    <title>agent.py:22-22 - Class-level _lock serializes all TransformationTracker...</title>
    <description><![CDATA[
### Location: nufi/agent.py:22-22

Class-level `_lock` serializes all `TransformationTracker` instances globally, even when operating
on independent directories. An instance-level lock (e.g., `self._lock = threading.Lock()` in
`__init__`) would allow concurrent trackers to operate in parallel.

Additionally, `save_snapshot()` writes the CSV under lock but calls `self.log_transformation()`
outside the critical section. Between those two operations another thread could interleave, causing
the log order to diverge from the actual save order. Consider wrapping both the CSV write and the
log call in a single `with self._lock:` block to make `save_snapshot` atomic.

-     _lock = threading.Lock()
+     # Instance-level lock would reduce contention for independent trackers.
+     # If class-level serialization is intentional (e.g., global filesystem safety),
+     # ensure save_snapshot is atomic: wrap both CSV write + log_transformation
+     # in a single with self._lock block.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>test_agent.py:73-77 - The test expects ValueError, but the impute_dataframe...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:73-77

The test expects `ValueError`, but the `impute_dataframe` code path for an empty DataFrame with
`time_col="timestamp"` reaches the index-type check first. An empty DataFrame created via
`pd.DataFrame(columns=["timestamp", "signal"])` has `object`-dtype columns, so after
`set_index("timestamp")` the index is non-numeric, and the function raises `TypeError` (agent.py
line 195) before any `ValueError` can be raised. This test will fail.

Suggestion: either change the expected exception to `TypeError`, or use `(TypeError, ValueError)` to
be safe, and consider adding a comment explaining which code path triggers the error.

      def test_impute_dataframe_empty(self):
          """Edge case: empty DataFrame should raise or return gracefully."""
          empty_df = pd.DataFrame(columns=["timestamp", "signal"])
-         with self.assertRaises(ValueError):
+         with self.assertRaises(TypeError):
              impute_dataframe(empty_df, time_col="timestamp")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>test_agent.py:83-89 - The test only verifies that the result column is still al...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:83-89

The test only verifies that the result column is still all-NaN, but ignores the `diagnostics` dict
returned by `impute_dataframe`. When all values are NaN, the source code (agent.py lines 234-243)
sets `"stability_flags": ["NO_OBSERVATIONS"]` and `"snr_db": None`. Asserting on these would catch
regressions where the diagnostics path silently breaks or the function fails to handle this edge
case properly.

Suggestion: add assertions on the diagnostics dict, e.g., `self.assertIn("NO_OBSERVATIONS",
diagnostics["signal"]["stability_flags"])`.

          result_df, diagnostics = impute_dataframe(
              all_nan_df,
              time_col="timestamp",
              log_path=self.test_log,
              history_dir=self.test_history
          )
          self.assertTrue(result_df["signal"].isna().all())
+         self.assertIn("signal", diagnostics)
+         self.assertIn("NO_OBSERVATIONS", diagnostics["signal"]["stability_flags"])
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>13</id>
    <title>impute.py:121-122 - If all candidate n_f values fail with RuntimeError (e.g...</title>
    <description><![CDATA[
### Location: nufi/impute.py:121-122

If all candidate `n_f` values fail with RuntimeError (e.g., all SVD calls fail), `best_n_freq` and
`best_alpha` retain their initial defaults silently. No warning is emitted that GCV tuning failed
entirely. In `transform()`, `solve_tikhonov_nudft` (line 202) is called without error handling, so
the same underlying failure will crash at inference time with a cryptic PyTorch RuntimeError.
Consider detecting this case after the loop and either raising a clear error or falling back to a
robust default with a prominent warning.

+             if best_gcv == float('inf'):
+                 import warnings
+                 warnings.warn(f"All GCV candidates failed for column {col_idx}; using fallback parameters.")
              self.alphas_.append(best_alpha)
              self.n_frequencies_.append(best_n_freq)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>14</id>
    <title>impute.py:141-152 - perm_small returned by covariance_compensation is cap...</title>
    <description><![CDATA[
### Location: nufi/impute.py:141-152

`perm_small` returned by `covariance_compensation` is captured but never used. `self.perm_` is
unconditionally set to `np.arange(n_cols)` (identity permutation), which discards the actual LDL^T
permutation ordering. If any downstream code relies on `self.perm_` together with `self.lu_` for
solving linear systems or applying the compensation, it will produce incorrect results. Consider
expanding `perm_small` to the full column space similarly to how `lu_small` and `d_small` are
expanded, so that `self.perm_` reflects the correct permutation.

                  lu_small, d_small, perm_small = covariance_compensation(X_list, device=self.device)
                  
                  # Expand to full size
                  self.lu_ = np.eye(n_cols)
                  self.d_ = np.eye(n_cols)
+                 # Map perm_small indices back to full column space
                  self.perm_ = np.arange(n_cols)
+                 for i, c_i in enumerate(valid_cols):
+                     self.perm_[c_i] = valid_cols[perm_small[i]]
                  
                  # Map small matrices back to full size
                  for i, c_i in enumerate(valid_cols):
                      for j, c_j in enumerate(valid_cols):
                          self.lu_[c_i, c_j] = lu_small[i, j]
                          self.d_[c_i, c_j] = d_small[i, j]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>15</id>
    <title>impute.py:111-114 - The warning message always says "SVD failed", but the Ru...</title>
    <description><![CDATA[
### Location: nufi/impute.py:111-114

The warning message always says "SVD failed", but the `RuntimeError` may originate from
`optimize_alpha_gcv` (line 100-101) rather than from a direct SVD call. Consider making the message
more precise (e.g., "Candidate evaluation failed...") to aid debugging.

                  except RuntimeError as e:
                      import warnings
-                     warnings.warn(f"SVD failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
+                     warnings.warn(f"Candidate evaluation failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
                      continue
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>16</id>
    <title>torch_kernels.py:52-58 - Issue (medium): O(N²) memory via outer product.</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:52-58

**Issue (medium): O(N²) memory via outer product.**

The expression `t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)` allocates a dense `(n_valid × N)`
complex128 tensor. For long time series (e.g., N=100k), this requires ~150 GB of memory and will
cause an OOM error. 

**Suggestions:**
- Add a guard that falls back to a batched or iterative computation when `N` exceeds a threshold
(e.g., > 10,000).
- Alternatively, route large-N cases through `compute_Fast_ND_NUDFT` automatically.
- Consider using `torch.einsum` with a chunked approach or a memory-mapped implementation.

          f_k = torch.linspace(0, nyquist_frequency, N, dtype=torch.float64, device=dev)
  
          # Standard NUDFT: A[n,k] = exp(-2πi * t_n * f_k), then sum over n
          t_timestamps = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
          t_data_all = torch.tensor(v_data, dtype=torch.float64, device=dev)
+ 
+         # Guard against excessive memory for large N
+         MAX_MEM_N = 10_000
+         if N > MAX_MEM_N:
+             import warnings
+             warnings.warn(
+                 f"N={N} is large; compute_ND_NUDFT may consume excessive memory. "
+                 f"Consider using compute_Fast_ND_NUDFT."
+             )
+ 
          exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
          summation = torch.sum(t_data_all.to(torch.complex128).unsqueeze(1) * torch.exp(exponent), dim=0)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>17</id>
    <title>torch_kernels.py:126-134 - Issue (high): NaN-to-zero replacement can produce a sin...</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:126-134

**Issue (high): NaN-to-zero replacement can produce a singular covariance matrix, leading to a
potential crash or meaningless LDL decomposition.**

`np.nan_to_num(covariance_matrix, nan=0.0)` replaces NaN entries (arising from zero-variance
columns) with zeros. This can make the matrix singular or non-positive-definite, which may cause
`scipy.linalg.ldl()` to raise a `LinAlgError` or return degenerate D factors. The existing warning
is passive and does not prevent the failure.

Additionally, concatenating real and imaginary parts doubles the feature dimension (`2M` columns for
`M` signals). The caller in `nufi/impute.py` only maps the top-left `[0:M, 0:M]` sub-block, so the
imaginary-part correlations are computed but silently discarded.

**Suggestions:**
- Before `nan_to_num`, check if NaN columns exist and drop them (or use a regularization like `cov +
eps * I`).
- Wrap `scipy.linalg.ldl()` in a try/except and fall back to a regularized Cholesky or
eigendecomposition.
- Consider whether concatenating real and imaginary parts is the intended design; if only real-part
correlation is needed, skip the imaginary concatenation to avoid the doubled dimension.

-     if np.any(np.isnan(covariance_matrix)):
+     nan_mask = np.any(np.isnan(covariance_matrix), axis=0)
+     if np.any(nan_mask):
          import warnings
-         warnings.warn("Covariance matrix contains NaN entries; degenerate columns detected.")
+         n_nan = nan_mask.sum()
+         warnings.warn(f"Covariance matrix contains NaN entries in {n_nan} columns; degenerate columns detected. Applying regularization.")
+         # Drop degenerate rows/columns instead of zero-filling
+         valid_idx = np.where(~nan_mask)[0]
+         if len(valid_idx) == 0:
+             raise ValueError("All columns are degenerate; cannot compute covariance compensation.")
+         covariance_matrix = covariance_matrix[np.ix_(valid_idx, valid_idx)]
      
-     covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0)
+     # Regularize to ensure positive-definiteness for LDL^T
+     eps = 1e-10
+     covariance_matrix = covariance_matrix + eps * np.eye(covariance_matrix.shape[0])
  
      # Step 4: Perform LDL^T decomposition
      # LDL^T factorizes A = P * L * D * L^T
+     try:
      lu, d, perm = scipy.linalg.ldl(covariance_matrix)
+     except np.linalg.LinAlgError:
+         raise ValueError("LDL decomposition failed; covariance matrix may be singular.")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>18</id>
    <title>torch_kernels.py:87-91 - Issue (medium): np.interp requires monotonically incr...</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:87-91

**Issue (medium): `np.interp` requires monotonically increasing `xp`, but `v_timestamps` is not
guaranteed to be sorted.**

If the input timestamps are unsorted (e.g., a DataFrame with a non-monotonic time index),
`np.interp` will produce incorrect interpolated values without any error or warning. This is a
silent correctness bug.

**Suggestion:** Sort `v_timestamps` and `v_data` together before calling `np.interp`, or validate
that `v_timestamps` is monotonically increasing and raise an informative error if not.

          # Generate uniform grid using min/max of valid timestamps
          t_min, t_max = np.min(v_timestamps), np.max(v_timestamps)
          uniform_grid = np.linspace(t_min, t_max, N)
-         # Interpolate onto uniform grid
+         # Ensure timestamps are sorted for np.interp (requires monotonic increasing)
+         if not np.all(np.diff(v_timestamps) >= 0):
+             sort_idx = np.argsort(v_timestamps)
+             v_timestamps = v_timestamps[sort_idx]
+             v_data = v_data[sort_idx]
          uniform_data = np.interp(uniform_grid, v_timestamps, v_data)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>19</id>
    <title>test_imputer.py:144-148 - Critical bug</title>
    <description><![CDATA[
### Location: tests/test_imputer.py:144-148

**Critical bug**: `random_state=42` causes the RNG to be seeded identically on every `transform()`
call (line 239 of `impute.py` creates a fresh `np.random.RandomState(42)` each time). Both
`X_filled_1` and `X_filled_2` will receive identical noise, so the assertions at lines 160–161 will
**fail**.

**Suggestion**: Either remove `random_state=42` from this test to restore non-deterministic
behavior, or restructure into two separate tests — one verifying reproducibility with a fixed seed,
another verifying non-determinism without a seed.

-     imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=42)
+     # Remove random_state to preserve non-deterministic behavior
+     imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False)
      imputer.fit(X)
      
      X_filled_1 = imputer.transform(X, stochastic=True, stochastic_scale=1.5)
      X_filled_2 = imputer.transform(X, stochastic=True, stochastic_scale=1.5)
]]></description>
  </task>
</tasklist>
```