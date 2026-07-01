```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>pyproject.toml:20-21 - The gpu optional dependency is redundant: torch&gt;=2.0.0...</title>
    <description><![CDATA[
### Location: pyproject.toml:20-21

The `gpu` optional dependency is redundant: `torch>=2.0.0` is already listed as a core dependency.
Installing `nufi[gpu]` provides no additional packages. If the intent is to make torch optional
(CPU-only mode), torch should be removed from core `dependencies` and placed only here. Otherwise,
this extra should either contain actual GPU-specific packages (e.g., CUDA toolkit bindings) or be
removed entirely.

  [project.optional-dependencies]
+ # Consider: if torch should be optional for CPU-only mode, move it from `dependencies` to this extra.
+ # Otherwise, remove this extra or list actual GPU-specific packages.
  gpu = ["torch>=2.0.0"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>agent.py:78-88 - Backward compatibility bug</title>
    <description><![CDATA[
### Location: nufi/agent.py:78-88

**Backward compatibility bug**: The new version ID format (`ver_<ts>_<uuid8>`) breaks parsing of
legacy snapshots. Old filenames like `ver_1234567890_pre_infill.csv` split into 4 parts (ver, ts,
"pre", "infill.csv"), which now matches `len(parts) >= 4` and incorrectly produces
`version_id="ver_1234567890_pre"` and `step_name="infill"` instead of `"ver_1234567890"` /
`"pre_infill"`. This causes `revert_to_version` to fail for any pre-upgrade snapshots, since the log
file references the old-style version ID. Consider detecting the UUID segment (e.g., via a regex
like `re.match(r'^[a-f0-9]{8}$', parts[2])`) to distinguish old vs new formats, or use a delimiter
other than underscore in the version ID.

                  for f in files:
                      parts = f.split("_")
-                     if len(parts) >= 4:
+                     # New format: ver_<ts>_<uuid8>_<step>.csv (≥5 parts with hex UUID)
+                     if len(parts) >= 4 and re.match(r'^[a-f0-9]{8}$', parts[2]):
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
    <id>3</id>
    <title>setup.py:66-80 - If tempfile.mkdtemp() raises an exception (e.g., Permis...</title>
    <description><![CDATA[
### Location: setup.py:66-80

If `tempfile.mkdtemp()` raises an exception (e.g., PermissionError), `tmpdir` is never assigned. The
`finally` block then raises a `NameError`, which masks the original failure and produces a confusing
error message. Move `mkdtemp()` inside the `try` block so the `finally` block only executes cleanup
when `tmpdir` was successfully created.

          has_openmp = False
-         tmpdir = tempfile.mkdtemp()
+         tmpdir = None
          try:
+             tmpdir = tempfile.mkdtemp()
              test_file = os.path.join(tmpdir, "test.c")
              with open(test_file, "w") as f:
                  f.write("#include <omp.h>\nint main(void) { return omp_get_num_threads(); }\n")
              cc = os.environ.get("CC", "cc")
              cmd = [cc, "-fopenmp", test_file, "-o", os.path.join(tmpdir, "test")]
              res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
              if res.returncode == 0:
                  has_openmp = True
          except Exception:
              pass
          finally:
+             if tmpdir is not None:
              shutil.rmtree(tmpdir)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>test_agent.py:75-77 - This test expects ValueError, but the actual implementa...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:75-77

This test expects `ValueError`, but the actual implementation raises `TypeError` for an empty
DataFrame. When `pd.DataFrame(columns=["timestamp", "signal"])` is created, the empty "timestamp"
column has dtype `object`. After `set_index("timestamp")`, the index dtype remains `object`, which
fails the `is_numeric_dtype` check at agent.py:194 and raises `TypeError`. Change to
`assertRaises(TypeError)` or add `TypeError` to the tuple.

          empty_df = pd.DataFrame(columns=["timestamp", "signal"])
-         with self.assertRaises(ValueError):
+         with self.assertRaises(TypeError):
              impute_dataframe(empty_df, time_col="timestamp")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>5</id>
    <title>setup.py:43-51 - The macOS OpenMP detection only checks two hardcoded Home...</title>
    <description><![CDATA[
### Location: setup.py:43-51

The macOS OpenMP detection only checks two hardcoded Homebrew paths. If libomp is installed
elsewhere (e.g., via Conda, MacPorts, or a custom prefix), OpenMP will be silently disabled,
potentially degrading performance. Consider also checking a user-configurable environment variable
(e.g., `LIBOMP_ROOT`) or attempting to use `pkg-config --cflags --libs libomp` for a more robust
detection mechanism.

          libomp_candidates = [
              "/opt/homebrew/opt/libomp",   # Apple Silicon Homebrew
              "/usr/local/opt/libomp",       # Intel Homebrew
          ]
+         # Also support user-specified path via environment variable
+         env_libomp = os.environ.get("LIBOMP_ROOT")
+         if env_libomp:
+             libomp_candidates.insert(0, env_libomp)
          libomp_path = None
          for candidate in libomp_candidates:
              if os.path.isdir(candidate):
                  libomp_path = candidate
                  break
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>6</id>
    <title>test_agent.py:93-100 - pd.testing.assert_frame_equal expects near-exact equali...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:93-100

`pd.testing.assert_frame_equal` expects near-exact equality, but `impute_dataframe` performs NUDFT
fitting/reconstruction which introduces floating-point differences even when there are no NaNs. This
test may fail due to numerical precision rather than a real bug. Consider using `assert_frame_equal`
with relaxed tolerances (e.g., `rtol=1e-3, atol=1e-3`) or checking that the imputed values are close
to the originals with `np.allclose`.

          clean_df = pd.DataFrame({"timestamp": [1.0, 2.0, 3.0], "signal": [10.0, 20.0, 30.0]})
          result_df, _ = impute_dataframe(
              clean_df,
              time_col="timestamp",
              log_path=self.test_log,
              history_dir=self.test_history
          )
-         pd.testing.assert_frame_equal(clean_df, result_df)
+         pd.testing.assert_frame_equal(clean_df, result_df, atol=1e-2)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>7</id>
    <title>wrappers.py:40-43 - When keep_time_col=True, the time column is re-added vi...</title>
    <description><![CDATA[
### Location: nufi/wrappers.py:40-43

When `keep_time_col=True`, the time column is re-added via `pd_df[time_col] = time_values` after
`set_index(time_col)`. This appends the column to the end of the DataFrame, altering the original
column order (the time column moves from its original position to the last position). Downstream
code that assumes a specific column layout — e.g., column-indexed access or positional slicing — may
break silently. Consider using `pd_df.insert(loc, time_col, time_values)` to preserve the original
column position.

          if keep_time_col:
              time_values = pd_df[time_col].copy()
+             col_pos = pd_df.columns.get_loc(time_col)
              pd_df = pd_df.set_index(time_col)
-             pd_df[time_col] = time_values
+             pd_df.insert(col_pos, time_col, time_values)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>agent.py:22-22 - Class-level lock causes unnecessary cross-instance contention</title>
    <description><![CDATA[
### Location: nufi/agent.py:22-22

**Class-level lock causes unnecessary cross-instance contention**: `_lock` is a class attribute
shared by all `TransformationTracker` instances, even those managing completely independent
directories. Two unrelated trackers (e.g., one for `log_dir_a` and one for `log_dir_b`) will
serialize against each other, reducing throughput in multi-threaded scenarios. Consider making the
lock per-instance by moving it to `__init__`: `self._lock = threading.Lock()`.

-     _lock = threading.Lock()
+     # Remove class-level _lock; use per-instance lock initialized in __init__
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>agent.py:25-26 - Weak path traversal check</title>
    <description><![CDATA[
### Location: nufi/agent.py:25-26

**Weak path traversal check**: Checking only for `".."` after `os.path.normpath` does not prevent an
attacker from providing an absolute path like `/etc/passwd` as `log_path`. While
`TransformationTracker` is typically called internally, `log_path` and `history_dir` are exposed
through `impute_dataframe`'s public API. Consider either (a) restricting both paths to a whitelisted
base directory with `os.path.commonpath`, or (b) ensuring paths are relative-only and resolved
against a safe root.

-         if ".." in os.path.normpath(log_path) or ".." in os.path.normpath(history_dir):
-             raise ValueError("Path traversal detected in log_path or history_dir.")
+         safe_root = os.path.abspath(os.getcwd())
+         for path in (log_path, history_dir):
+             resolved = os.path.abspath(path)
+             if os.path.commonpath([safe_root, resolved]) != safe_root:
+                 raise ValueError(f"Path {path} is outside the allowed directory.")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>10</id>
    <title>agent.py:339-355 - New parameters not documented in docstring</title>
    <description><![CDATA[
### Location: nufi/agent.py:339-355

**New parameters not documented in docstring**: `solver`, `max_iter`, `tol`, and `device` were added
to `plot_diagnostics` but are missing from the function's docstring. This makes it unclear to users
what these parameters control and what valid values are. Additionally, `device` lacks a type hint
(should be `str = None` or `Optional[str] = None`).

  def plot_diagnostics(
      original_df: pd.DataFrame,
      infilled_df: pd.DataFrame,
      diagnostics: dict,
      time_col: str = None,
      columns: list = None,
      save_path: str = None,
      show_plot: bool = True,
      solver: str = 'direct',
      max_iter: int = 100,
      tol: float = 1e-5,
      device: str = None
  ):
      """
      Generates an interactive, publication-ready visualization of the infilling results.
      Plots both time-domain reconstructions and frequency-domain power spectrum densities.
+ 
+     Parameters
+     ----------
+     ...
+     solver : str, default='direct'
+         Linear system solver for PSD computation ('direct' or 'cg').
+     max_iter : int, default=100
+         Maximum iterations for CG solver.
+     tol : float, default=1e-5
+         Tolerance for CG solver convergence.
+     device : str, optional
+         Hardware accelerator device for PSD computation.
      """
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>test_imputer.py:144-144 - The random_state=42 added to the imputer constructor ma...</title>
    <description><![CDATA[
### Location: tests/test_imputer.py:144-144

The `random_state=42` added to the imputer constructor makes every `transform(stochastic=True)` call
deterministic: `np.random.RandomState(42)` produces the same noise sequence on each call. This
contradicts the test's purpose of verifying non-deterministic stochastic imputation. The assertions
at lines 160-161 (`X_filled_1[1,0] != X_filled_2[1,0]` etc.) will fail because the two calls produce
identical filled arrays. Fix: remove `random_state=42` from this test, since this test specifically
validates non-deterministic behavior. Use `random_state` only in tests that need reproducibility
(like `test_stochastic_imputation_multicol`).

-     imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=42)
+     imputer = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>test_imputer.py:190-196 - The same imputer instance is reused across three consec...</title>
    <description><![CDATA[
### Location: tests/test_imputer.py:190-196

The same `imputer` instance is reused across three consecutive `fit_transform()` calls with data of
different shapes (3×2, 1×2, 3×1). While `fit()` currently resets all internal state (`self.lu_`,
`self.d_`, `self.perm_`, `self.alphas_`, `self.n_frequencies_`), this is fragile — a future state
variable that isn't reset in `fit()` could leak between scenarios. Each edge case should use a fresh
`NufiImputer` instance to guarantee isolation and make the test self-documenting.

-     imputer = NufiImputer(covariance_compensation=True)
-     X_filled = imputer.fit_transform(X_all_nan)
+     imputer1 = NufiImputer(covariance_compensation=True)
+     X_filled = imputer1.fit_transform(X_all_nan)
      assert np.isnan(X_filled[:, 1]).all()  # column with all NaNs remains NaN or handles gracefully
      
      # 2. Single-row input
      X_single_row = np.array([[1.0, np.nan]], dtype=np.float64)
-     X_filled_row = imputer.fit_transform(X_single_row)
+     imputer2 = NufiImputer(covariance_compensation=True)
+     X_filled_row = imputer2.fit_transform(X_single_row)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>13</id>
    <title>test_agent.py:79-89 - This test asserts that an all-NaN column remains all-NaN ...</title>
    <description><![CDATA[
### Location: tests/test_agent.py:79-89

This test asserts that an all-NaN column remains all-NaN after imputation, but the
`NufiImputer.transform()` behavior for columns with zero valid observations is not explicitly
defined in the implementation. If the imputer applies any default fill value or raises an exception,
this assertion will fail or give a false signal. Consider verifying the behavior by inspecting the
`NufiImputer.transform` logic, or document this as an expected contract rather than an
implementation coincidence.

      def test_impute_dataframe_all_nan(self):
          """Edge case: column with all NaN values."""
          all_nan_df = self.df.copy()
          all_nan_df["signal"] = np.nan
          result_df, diagnostics = impute_dataframe(
              all_nan_df,
              time_col="timestamp",
              log_path=self.test_log,
              history_dir=self.test_history
          )
+         # When all values are NaN, the imputer cannot fit; expect NaN output or error
          self.assertTrue(result_df["signal"].isna().all())
+         self.assertIn("NO_OBSERVATIONS", diagnostics.get("signal", {}).get("stability_flags", []))
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>14</id>
    <title>agent.py:51-66 - Atomicity gap between CSV write and log write</title>
    <description><![CDATA[
### Location: nufi/agent.py:51-66

**Atomicity gap between CSV write and log write**: `save_snapshot` writes the CSV inside the lock
but calls `self.log_transformation(log_entry)` outside it. If `log_transformation` fails (or another
thread reads between these two operations), the CSV file exists on disk without a corresponding log
entry, creating an orphaned snapshot. Consider moving the log write inside the same lock block to
ensure atomicity of the snapshot+log operation.

          with self._lock:
              try:
                  df.to_csv(filepath, index=True)
              except Exception as e:
                  raise TransformationLoggingError(f"Failed to save data snapshot {filepath}: {e}")
              
          log_entry = {
              "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
              "event": "snapshot_saved",
              "version_id": version_id,
              "step_name": step_name,
              "columns": list(df.columns),
              "shape": df.shape,
              "filepath": filepath
          }
-         self.log_transformation(log_entry)
+             # Write log within the same lock to ensure atomicity
+             try:
+                 with open(self.log_path, "a", encoding="utf-8") as f:
+                     f.write(json.dumps(log_entry) + "\n")
+             except Exception as e:
+                 # Rollback: remove orphaned CSV on log failure
+                 try:
+                     os.remove(filepath)
+                 except OSError:
+                     pass
+                 raise TransformationLoggingError(f"Failed to write to transformation log: {e}")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>15</id>
    <title>agent.py:194-198 - Breaking API change without migration path</title>
    <description><![CDATA[
### Location: nufi/agent.py:194-198

**Breaking API change without migration path**: The new `is_numeric_dtype` check in
`impute_dataframe` and `plot_diagnostics` will raise `TypeError` for users who previously passed
DataFrames with string-based or datetime-based indices (e.g., `"2024-01-01"` strings). While the
requirement for numeric timestamps is technically correct, this is a hard breaking change. Consider
either (a) attempting automatic conversion via `pd.to_numeric(df_copy.index, errors='raise')` before
failing, or (b) documenting this as a breaking change with clear migration guidance in the error
message.

      if not pd.api.types.is_numeric_dtype(df_copy.index):
+         try:
+             # Attempt conversion for datetime-like or string timestamps
+             numeric_idx = pd.to_numeric(df_copy.index, errors='coerce')
+             if numeric_idx.isna().any():
+                 raise ValueError("Index contains non-convertible values")
+             df_copy.index = numeric_idx.astype(np.float64)
+         except Exception:
          raise TypeError(
              f"DataFrame index must be numeric (timestamps). "
-             f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col`."
+                 f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col` "
+                 f"or ensure your index contains numeric values."
          )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>16</id>
    <title>impute.py:239-239 - np.random.RandomState() only accepts an integer seed or...</title>
    <description><![CDATA[
### Location: nufi/impute.py:239-239

`np.random.RandomState()` only accepts an integer seed or `None`. If a user passes a
`numpy.random.RandomState` instance (standard scikit-learn convention), this will raise `ValueError:
Input must be an integer`. Use `sklearn.utils.check_random_state(self.random_state)` to handle
`int`, `RandomState`, and `None` correctly.

-                     rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
+                     from sklearn.utils import check_random_state
+                     rng = check_random_state(self.random_state)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>17</id>
    <title>impute.py:141-146 - The perm_small vector returned by covariance_compensat...</title>
    <description><![CDATA[
### Location: nufi/impute.py:141-146

The `perm_small` vector returned by `covariance_compensation()` (the LDLᵀ permutation) is discarded
and replaced with `np.arange(n_cols)`. While `self.perm_` is not currently consumed in `transform`,
the stored attribute becomes incorrect, which could cause subtle bugs if future code relies on it
for column reordering. Consider mapping `perm_small` through `valid_cols` to preserve the correct
permutation.

                  lu_small, d_small, perm_small = covariance_compensation(X_list, device=self.device)
                  
                  # Expand to full size
                  self.lu_ = np.eye(n_cols)
                  self.d_ = np.eye(n_cols)
+                 # Map perm_small indices through valid_cols, leaving excluded columns at identity
                  self.perm_ = np.arange(n_cols)
+                 for i, c_i in enumerate(valid_cols):
+                     self.perm_[c_i] = valid_cols[perm_small[i]]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>18</id>
    <title>impute.py:97-111 - The try-except block wraps both SVD and subsequent tensor...</title>
    <description><![CDATA[
### Location: nufi/impute.py:97-111

The try-except block wraps both SVD and subsequent tensor operations (`matmul`, `sum`, `to`). Only
`torch.linalg.svd` and the internal SVD inside `optimize_alpha_gcv` are expected to raise
`RuntimeError`. Wrapping the matrix multiplications and GCV scoring in the same try block could
silently swallow unrelated `RuntimeError` bugs (e.g., shape mismatches, out-of-memory). Consider
narrowing the try block to only the SVD calls.

                  try:
                      if self.alpha == 'auto':
                          candidate_alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
                          opt_alpha, U, S, y_tilde, y_norm_sq = optimize_alpha_gcv(
                              A, t_data, candidate_alphas, return_svd=True
                          )
                      else:
                          opt_alpha = self.alpha if self.alpha is not None else 1e-4
                          U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                          y_complex = t_data.to(torch.complex128)
                          y_tilde = torch.matmul(U.adjoint(), y_complex)
                          y_norm_sq = torch.sum(torch.abs(y_complex) ** 2)
+                 except RuntimeError as e:
+                     import warnings
+                     warnings.warn(f"SVD failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
+                     continue
                      
                      score = compute_gcv_from_svd(S, y_tilde, y_norm_sq, opt_alpha, N_val)
-                 except RuntimeError as e:
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>19</id>
    <title>impute.py:121-122 - If every candidate n_f fails SVD for a column (all iter...</title>
    <description><![CDATA[
### Location: nufi/impute.py:121-122

If every candidate `n_f` fails SVD for a column (all iterations hit `continue`), `best_gcv` remains
`float('inf')` and the fallback alpha/freq are appended silently. Consider adding a warning when GCV
tuning produced no valid score, so users know the imputer fell back to untuned defaults for this
column.

+             if best_gcv == float('inf'):
+                 import warnings
+                 warnings.warn(
+                     f"All GCV candidates failed for column {col_idx}. "
+                     f"Using fallback alpha={best_alpha}, n_freq={best_n_freq}."
+                 )
              self.alphas_.append(best_alpha)
              self.n_frequencies_.append(best_n_freq)
]]></description>
  </task>
</tasklist>
```