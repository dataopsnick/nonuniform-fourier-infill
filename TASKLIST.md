<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>pyproject.toml:21-21 - The gpu optional-dependency is redundant: torch&gt;=2.0.0...</title>
    <description><![CDATA[
### Location: pyproject.toml:21-21

The `gpu` optional-dependency is redundant: `torch>=2.0.0` is already a core dependency (line 18).
Installing `nufi[gpu]` provides no additional value over a plain `nufi` install. Consider either:
(a) removing `torch` from core `dependencies` and keeping it only in the `gpu` extra so torch
becomes truly optional, or (b) adding GPU-specific packages to the extra (e.g.,
`nvidia-cublas-cu12`, pinning to a known-good torch+CUDA index, etc.).

- gpu = ["torch>=2.0.0"]
+ # Option A: make torch optional
+ # Remove "torch>=2.0.0" from [project] dependencies above
+ # Option B: add GPU-specific deps
+ gpu = ["torch>=2.0.0", "nvidia-cublas-cu12"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>benchmark.py:98-98 - DataFrame.fillna(method='ffill') and .fillna(method='b...</title>
    <description><![CDATA[
### Location: tests/benchmark.py:98-98

`DataFrame.fillna(method='ffill')` and `.fillna(method='bfill')` were deprecated in pandas 1.5.0 in
favor of `.ffill()` / `.bfill()`. The previous code used the idiomatic `.ffill().bfill()` which is
forward-compatible — this change regresses to a deprecated API and will emit `FutureWarning` on
pandas ≥ 1.5.0.

-         spline_infilled = spline_infilled.interpolate(method='linear', axis=0).fillna(method='ffill').fillna(method='bfill')
+         spline_infilled = spline_infilled.interpolate(method='linear', axis=0).ffill().bfill()
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>setup.py:34-34 - Typo in environment variable name: the project is named "...</title>
    <description><![CDATA[
### Location: setup.py:34-34

Typo in environment variable name: the project is named "nufi", but the variable uses "NUIFI" (extra
'I'). Users referencing the project name would expect `NUFI_NO_OPENMP`. Consider renaming to
`NUFI_NO_OPENMP` to avoid confusion.

- if os.environ.get("NUIFI_NO_OPENMP"):
+ if os.environ.get("NUFI_NO_OPENMP"):
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>setup.py:43-46 - The libomp search only checks two Homebrew paths. Users w...</title>
    <description><![CDATA[
### Location: setup.py:43-46

The libomp search only checks two Homebrew paths. Users who install libomp via MacPorts
(`/opt/local/lib/libomp`), conda, or a custom build will have OpenMP silently disabled with no clear
path to enable it. Consider adding more fallback paths, or allowing the user to override via an
environment variable like `LIBOMP_PATH`.

          libomp_candidates = [
+             os.environ.get("LIBOMP_PATH", ""),
              "/opt/homebrew/opt/libomp",   # Apple Silicon Homebrew
              "/usr/local/opt/libomp",       # Intel Homebrew
+             "/opt/local/lib/libomp",       # MacPorts
          ]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>5</id>
    <title>setup.py:57-57 - This warning doesn't mention the NUIFI_NO_OPENMP (or co...</title>
    <description><![CDATA[
### Location: setup.py:57-57

This warning doesn't mention the `NUIFI_NO_OPENMP` (or corrected `NUFI_NO_OPENMP`) toggle. Users
with libomp in a non-standard location can't easily suppress this message. Consider adding a hint:
`Set NUFI_NO_OPENMP=1 to suppress this warning.`

-             print("WARNING: OpenMP not found. Install libomp via 'brew install libomp' for better performance.")
+             print("WARNING: OpenMP not found. Install libomp via 'brew install libomp' for better performance, or set NUFI_NO_OPENMP=1 to suppress this warning.")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>6</id>
    <title>test_agent.py:73-77 - impute_dataframe does not raise ValueError for empty ...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:73-77

`impute_dataframe` does not raise `ValueError` for empty DataFrames. Tracing through the
implementation: `NufiImputer.fit()` handles `N_val == 0` by skipping to default parameters, and
`transform()` handles `len(v_data) == 0` by copying the empty column data. The function returns
gracefully with a `"NO_OBSERVATIONS"` diagnostic flag. This test will fail. Either update the test
to match the actual behavior (e.g., assert the result is an empty DataFrame with expected
diagnostics) or add an explicit empty-DataFrame guard in `impute_dataframe`.

      def test_impute_dataframe_empty(self):
-         """Edge case: empty DataFrame should raise or return gracefully."""
+         """Edge case: empty DataFrame should return gracefully with diagnostics."""
          empty_df = pd.DataFrame(columns=["timestamp", "signal"])
-         with self.assertRaises(ValueError):
-             impute_dataframe(empty_df, time_col="timestamp")
+         result_df, diagnostics = impute_dataframe(empty_df, time_col="timestamp")
+         self.assertTrue(result_df.empty)
+         self.assertIn("signal", diagnostics)
+         self.assertIn("NO_OBSERVATIONS", diagnostics["signal"]["stability_flags"])
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>7</id>
    <title>test_agent.py:4-4 - The json module is imported but never used in this test...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:4-4

The `json` module is imported but never used in this test file. This import should be removed to
keep the code clean.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>wrappers.py:39-45 - When keep_time_col=True, the time column is duplicated ...</title>
    <description><![CDATA[
### Location: nufi/wrappers.py:39-45

When `keep_time_col=True`, the time column is duplicated as both the DataFrame index and a regular
feature column. `NufiImputer.fit()` calls `X.to_numpy()` which includes ALL columns (including the
time column), so the time column's numerical values go through the full NUDFT pipeline — frequency
estimation, GCV tuning, and covariance compensation. This means: (1) the time column participates in
covariance estimation, which can distort the covariance matrix since timestamp values (e.g., Unix
epochs) are typically on a vastly different scale than signal values; (2) if the time column happens
to contain NaNs, the imputer will attempt to "infill" timestamps via Fourier reconstruction, which
is conceptually invalid; (3) even without NaNs, it wastes computation fitting a model for the time
column. Consider documenting this trade-off clearly or filtering out the time column from features
before passing to the imputer.

      if time_col is not None:
          if keep_time_col:
              time_values = pd_df[time_col].copy()
              pd_df = pd_df.set_index(time_col)
              pd_df[time_col] = time_values
+             # Note: the time column is now both the index and a feature column.
+             # The imputer will fit/transform it like any other column, which may
+             # distort covariance estimation if timestamp values differ in scale.
          else:
              pd_df = pd_df.set_index(time_col)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>wrappers.py:77-80 - When sort=False is passed and the MultiIndex time level...</title>
    <description><![CDATA[
### Location: nufi/wrappers.py:77-80

When `sort=False` is passed and the MultiIndex time level is not in chronological order,
`np.diff(v_timestamps)` inside `NufiImputer.fit()` and `transform()` will produce negative values
for decreasing timestamp pairs. The Nyquist frequency calculation uses `np.nanmin(p_n)`, which will
be negative (or zero), causing a fallback to `min_p = 1.0` and an incorrect Nyquist frequency. This
silently degrades imputation quality without any warning to the user. Consider either issuing a
warning when `sort=False` is used or validating that timestamps are monotonically increasing before
passing to the imputer.

      sort : bool, default True
          Whether to sort the index by time level to ensure proper chronological order.
+         Setting sort=False when timestamps are not already sorted may produce
+         incorrect results due to Nyquist frequency miscalculation.
          Note: This reorders the group's rows; the output index order will reflect the sorted
          order, not the original input order.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>10</id>
    <title>wrappers.py:56-58 - The second import cudf at this point is redundant — cu...</title>
    <description><![CDATA[
### Location: nufi/wrappers.py:56-58

The second `import cudf` at this point is redundant — `cudf` is already in scope from the try/except
block at the top of the function (line 23-27). Consider removing this duplicate import for clarity.

      if is_cudf:
-         import cudf
          return cudf.DataFrame.from_pandas(infilled_pd)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>wrappers.py:56-58 - Same redundant import cudf as in infill_dataframe — ...</title>
    <description><![CDATA[
### Location: nufi/wrappers.py:56-58

Same redundant `import cudf` as in `infill_dataframe` — `cudf` is already in scope from the
try/except block (line 82-86). Consider removing for consistency.

      if is_cudf:
-         import cudf
          return cudf.DataFrame.from_pandas(infilled_pd)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>agent.py:78-88 - Backward-incompatible parsing: Old snapshots (without UUI...</title>
    <description><![CDATA[
### Location: nufi/agent.py:78-88

Backward-incompatible parsing: Old snapshots (without UUID suffix) with underscore-containing step
names (e.g., 'pre_infill') will be parsed incorrectly. With old-format filename
`ver_1719859200000_pre_infill.csv`, parts = ['ver','1719859200000','pre','infill.csv'] (len>=4)
yields version_id='ver_1719859200000_pre' instead of 'ver_1719859200000'. This causes
`revert_to_version` to fail for previously saved snapshots. Consider using a more robust parsing
strategy: detect the new format by checking if the third underscore-separated token is a hex string
of length 8 (UUID), and fall back to the old two-part version_id otherwise.

                  for f in files:
                      parts = f.split("_")
-                     if len(parts) >= 4:
+                     # Detect new format: ver_{ts}_{uuid8}_{step_name}.csv
+                     if len(parts) >= 4 and len(parts[2]) == 8 and all(c in '0123456789abcdef' for c in parts[2]):
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
    <id>13</id>
    <title>impute.py:148-152 - CRITICAL: d_small is 1D — indexing with [i, j] will raise IndexError at runtime.</title>
    <description><![CDATA[
### Location: nufi/impute.py:148-152

**CRITICAL: `d_small` is 1D — indexing with `[i, j]` will raise `IndexError` at runtime.**

`scipy.linalg.ldl(covariance_matrix)` returns `d` as a 1D array of shape `(2*m,)` where `m =
len(X_list)`. This is because `flat_data` now concatenates real+imag parts (see `torch_kernels.py`
line 116), making the covariance matrix `(2m, 2m)`. When the expansion loop executes `d_small[i,
j]`, numpy raises `IndexError: too many indices for array`. 

Additionally, even if `d` were 2D, only the top-left `m×m` corner of `d_small` would be mapped (the
loop iterates over `valid_cols` of length `m`, not `2m`). The same dimension mismatch affects the
`lu_small` assignment below.

Suggestion: Either revert `flat_data` to use only real parts (matching the pre-change behavior), or
redesign the expansion logic to handle the `2m×2m` structure properly (e.g., map only the real-part
sub-block or restructure the return of `covariance_compensation`).

-                 # Map small matrices back to full size
+                 # NOTE: lu_small is (2*m, 2*m) but we only need the m×m real-part block.
+                 # Revisit covariance_compensation return or extract the relevant sub-block here.
+                 # For now: if d_small is 1D, populate diagonal only.
+                 if d_small.ndim == 1:
+                     for i, c_i in enumerate(valid_cols):
+                         self.d_[c_i, c_i] = d_small[i]
+                 else:
                  for i, c_i in enumerate(valid_cols):
                      for j, c_j in enumerate(valid_cols):
-                         self.lu_[c_i, c_j] = lu_small[i, j]
                          self.d_[c_i, c_j] = d_small[i, j]
+                 for i, c_i in enumerate(valid_cols):
+                     for j, c_j in enumerate(valid_cols):
+                         self.lu_[c_i, c_j] = lu_small[i, j]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>14</id>
    <title>impute.py:141-146 - HIGH: perm_small is captured but ignored — lu_small/d_small entries are in permuted order.</title>
    <description><![CDATA[
### Location: nufi/impute.py:141-146

**HIGH: `perm_small` is captured but ignored — `lu_small`/`d_small` entries are in permuted order.**

`scipy.linalg.ldl` returns `lu` and `d` in a permuted space: the decomposition satisfies `L @ D @
L.T = A[perm, :][:, perm]`. The expansion loop maps `lu_small[i, j]` → `self.lu_[valid_cols[i],
valid_cols[j]]`, but `lu_small[i, j]` actually corresponds to positions `valid_cols[perm_small[i]]`
and `valid_cols[perm_small[j]]` in the original column ordering. 

This means the entries in `self.lu_` and `self.d_` are placed at incorrect indices, and any
downstream use of these matrices (including the diagonal scaling in `transform()`) will read values
intended for a different column.

Suggestion: Store the actual permutation, or apply the inverse permutation when populating the
full-size matrices. If a full identity permutation is acceptable for the expanded matrices, at least
document why `perm_small` is discarded.

                  lu_small, d_small, perm_small = covariance_compensation(X_list, device=self.device)
                  
-                 # Expand to full size
+                 # Expand to full size; apply inverse permutation for correct column alignment
                  self.lu_ = np.eye(n_cols)
                  self.d_ = np.eye(n_cols)
-                 self.perm_ = np.arange(n_cols)
+                 inv_perm = np.argsort(perm_small)
+                 self.perm_ = np.arange(n_cols)  # or map perm_small into full space if needed
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>15</id>
    <title>impute.py:238-240 - HIGH: RandomState created inside the per-column loop — every column receives identical noise.</title>
    <description><![CDATA[
### Location: nufi/impute.py:238-240

**HIGH: `RandomState` created inside the per-column loop — every column receives identical noise.**

When `self.random_state` is an integer seed, `np.random.RandomState(self.random_state)` is
re-instantiated on every iteration with the same seed. This produces the identical sequence of
random numbers for each column, destroying statistical independence of the stochastic noise across
columns. The resulting joint imputation distribution will be distorted, potentially biasing
downstream analyses (e.g., correlations, PCA).

Suggestion: Instantiate the RNG once outside the `for col_idx` loop, and advance it independently
for each column (or draw all noise in one shot).

                      # Generate noise from posterior process scaled by uncertainty parameters
-                     rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
                      noise = rng.normal(0, stochastic_scale * residual_std, size=nan_mask.sum())
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>16</id>
    <title>agent.py:22-26 - The class-level _lock serializes all operations across ...</title>
    <description><![CDATA[
### Location: nufi/agent.py:22-26

The class-level `_lock` serializes all operations across ALL TransformationTracker instances, even
those targeting different history directories/log files. This creates unnecessary contention in
multi-threaded scenarios where independent trackers are active. Use a per-instance lock instead by
initializing `self._lock = threading.Lock()` in `__init__`, and removing the class-level `_lock`.
This preserves thread safety while allowing independent trackers to operate concurrently.

-     _lock = threading.Lock()
- 
      def __init__(self, log_path: str = "nufi_transformations.log", history_dir: str = ".nufi_history"):
          if ".." in os.path.normpath(log_path) or ".." in os.path.normpath(history_dir):
              raise ValueError("Path traversal detected in log_path or history_dir.")
+         self._lock = threading.Lock()
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>17</id>
    <title>agent.py:25-28 - The path traversal guard only checks for '..' after normp...</title>
    <description><![CDATA[
### Location: nufi/agent.py:25-28

The path traversal guard only checks for '..' after normpath, but can be bypassed with absolute
paths (e.g., '/etc/passwd'), symlinks, or UNC paths on Windows. Since `abspath` does not resolve
symlinks, a symlinked directory could point outside the intended area. Consider using
`os.path.realpath` to resolve symlinks and validate that the resulting path is within an allowed
base directory (e.g., the current working directory or a configured root).

-         if ".." in os.path.normpath(log_path) or ".." in os.path.normpath(history_dir):
-             raise ValueError("Path traversal detected in log_path or history_dir.")
-         self.log_path = os.path.abspath(log_path)
-         self.history_dir = os.path.abspath(history_dir)
+         self.log_path = os.path.realpath(log_path)
+         self.history_dir = os.path.realpath(history_dir)
+         cwd = os.path.realpath(os.getcwd())
+         if not self.log_path.startswith(cwd + os.sep) and self.log_path != cwd:
+             raise ValueError(f"log_path must be within current working directory: {log_path}")
+         if not self.history_dir.startswith(cwd + os.sep) and self.history_dir != cwd:
+             raise ValueError(f"history_dir must be within current working directory: {history_dir}")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>18</id>
    <title>impute.py:179-181 - HIGH: RNG should be created here, before the column loo...</title>
    <description><![CDATA[
### Location: nufi/impute.py:179-181

**HIGH: RNG should be created here, before the column loop, to ensure independent noise per
column.**

Currently the `RandomState` is created inside the loop body (line 239), causing every column to
receive identical random numbers. Move the RNG creation here so each column draws from a single
advancing sequence.

          infilled_data = np.zeros_like(X_data)
+         
+         rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
          
          for col_idx in range(X_data.shape[1]):
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>19</id>
    <title>impute.py:121-122 - MEDIUM: No warning when all candidate frequencies fail SVD.</title>
    <description><![CDATA[
### Location: nufi/impute.py:121-122

**MEDIUM: No warning when all candidate frequencies fail SVD.**

If every `n_f` in `candidates` triggers a `RuntimeError` during SVD, the loop exits with
`best_n_freq = candidates[0]` and `best_alpha = 1e-4` (the initial defaults). These are then
appended to `self.n_frequencies_` and `self.alphas_` and used later in `transform()`. While the
defaults are reasonable, the column is effectively untuned and the user has no indication that all
GCV candidates failed for that column. A warning would help users diagnose problematic input data.

+             if best_gcv == float('inf'):
+                 import warnings
+                 warnings.warn(
+                     f"All GCV candidates failed SVD for column {col_idx}. "
+                     f"Using fallback n_f={best_n_freq}, alpha={best_alpha}."
+                 )
              self.alphas_.append(best_alpha)
              self.n_frequencies_.append(best_n_freq)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>20</id>
    <title>test_covariance.py:26-29 - scipy.interpolate.interp1d has been deprecated since sc...</title>
    <description><![CDATA[
### Location: tests/test_covariance.py:26-29

`scipy.interpolate.interp1d` has been deprecated since scipy 1.10.0. Consider using `numpy.interp`
(already imported) for a simpler, dependency-free linear interpolation baseline:

```python
linear_fill = np.interp(t, t[valid], signal[valid])
```

`np.interp` performs the same linear interpolation and avoids deprecation warnings during test runs.

      # Task 24: Baseline comparison with linear interpolation
-     from scipy.interpolate import interp1d
      valid = ~np.isnan(signal)
-     linear_fill = interp1d(t[valid], signal[valid], kind='linear', fill_value='extrapolate')(t)
+     linear_fill = np.interp(t, t[valid], signal[valid])
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>21</id>
    <title>test_covariance.py:59-60 - The covariance thresholds were tightened from rtol=1e-1,...</title>
    <description><![CDATA[
### Location: tests/test_covariance.py:59-60

The covariance thresholds were tightened from `rtol=1e-1, atol=1e-1` to `rtol=1e-2, atol=1e-2` — a
10× stricter requirement. While the `NufiImputer` with `covariance_compensation=True` may currently
satisfy this, such tight tolerances are fragile: small numerical changes in the underlying algorithm
(e.g., PyTorch version, floating-point precision, solver tweaks) could cause this test to break
without indicating a real regression.

Consider relaxing to `rtol=5e-2, atol=5e-2` or keeping `rtol=1e-1` to provide a more robust safety
margin while still verifying covariance preservation.

-     # Task 23: Tighten covariance preservation thresholds to rtol=1e-2, atol=1e-2
-     np.testing.assert_allclose(filled_cov, original_cov, rtol=1e-2, atol=1e-2)
+     # Verify that the filled covariance is close to the original covariance structure
+     np.testing.assert_allclose(filled_cov, original_cov, rtol=5e-2, atol=5e-2)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>22</id>
    <title>test_imputer.py:144-161 - Passing random_state=42 creates a fresh RandomState(42...</title>
    <description><![CDATA[
### Location: tests/test_imputer.py:144-161

Passing `random_state=42` creates a fresh `RandomState(42)` inside each `transform()` call, which
always starts from the same seed. This makes X_filled_1 and X_filled_2 identical, breaking the
stochastic non-determinism assertions at lines 160-161. To preserve non-determinism while being
explicit, either remove `random_state` here, or test reproducibility separately (e.g., compare two
imputers with the same seed).

-     imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=42)
+     imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False)
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
]]></description>
  </task>
</tasklist>
```