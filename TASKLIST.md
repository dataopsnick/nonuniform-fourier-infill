<tasklist>
  <task status="COMPLETED">
    <id>1</id>
    <title>setup.py:11-16 - OpenMP silently disabled on macOS</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:11-16

**OpenMP silently disabled on macOS** — When `sys.platform == 'darwin'`, OpenMP flags are set to empty lists without any warning to the user. This silently degrades performance. Modern macOS with Homebrew's `libomp` (`brew install libomp`) can support OpenMP via flags like `-Xpreprocessor -fopenmp` and `-lomp`. Consider detecting `libomp` availability (e.g., via `subprocess` or checking known paths like `/opt/homebrew/opt/libomp` or `/usr/local/opt/libomp`) and emitting a `print()` warning when OpenMP is unavailable so users are informed about the performance impact.

```python
  if sys.platform == "win32":
      ext_compiler_args = ["/openmp"]
  elif sys.platform == "darwin":
      # On macOS, standard clang doesn't support -fopenmp without extra config.
```

* ```python
  # We will try compiling without OpenMP, or user can set brew paths.
  ```
* ```python
  # Try to detect Homebrew libomp; fall back to no OpenMP with a warning.
  ```
* ```python
  import glob
  ```
* ```python
  libomp_candidates = [
  ```
* ```python
      "/opt/homebrew/opt/libomp",   # Apple Silicon Homebrew
  ```
* ```python
      "/usr/local/opt/libomp",       # Intel Homebrew
  ```
* ```python
  ]
  ```
* ```python
  libomp_path = None
  ```
* ```python
  for candidate in libomp_candidates:
  ```
* ```python
      if os.path.isdir(candidate):
  ```
* ```python
          libomp_path = candidate
  ```
* ```python
          break
  ```
* ```python
  if libomp_path:
  ```
* ```python
      ext_compiler_args = ["-Xpreprocessor", "-fopenmp"]
  ```
* ```python
      ext_linker_args = ["-L" + os.path.join(libomp_path, "lib"), "-lomp"]
  ```
* ```python
      ext_include = os.path.join(libomp_path, "include")
  ```
* ```python
      # Note: include_dirs in Extension must also be updated.
  ```
* ```python
  else:
  ```
* ```python
      print("WARNING: OpenMP not found. Install libomp via 'brew install libomp' for better performance.")
  ext_compiler_args = []
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>2</id>
    <title>test_agent.py:80-80 - Missing import json at the top of the file. The test_t...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:80-80

Missing `import json` at the top of the file. The `test_tracker_logging` method calls `json.loads()` on line 84, but the `json` module is never imported. This will raise a `NameError: name 'json' is not defined` at runtime, breaking the entire test suite.

* # Add at the top of the file:
* # import json
* ```python
      logged_data = json.loads(lines[0])
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>3</id>
    <title>test_agent.py:68-69 - Fragile assertion: self.assertEqual(len(csv_files), 2) ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:68-69

Fragile assertion: `self.assertEqual(len(csv_files), 2)` assumes exactly two CSV files exist in the history directory. If a previous test run left residual files or if `impute_dataframe` writes additional snapshots in the future, this test will fail flakily. Consider asserting that at least the expected files exist (e.g., check for 'pre_infill' and 'post_infill' substrings in filenames) instead of relying on an exact count.

```python
      csv_files = [f for f in files if f.endswith(".csv")]
```

* ```python
      self.assertEqual(len(csv_files), 2)  # pre_infill and post_infill
  ```
* ```python
      # Check that pre_infill and post_infill snapshots both exist
  ```
* ```python
      self.assertTrue(any("pre_infill" in f for f in csv_files))
  ```
* ```python
      self.assertTrue(any("post_infill" in f for f in csv_files))
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>4</id>
    <title>test_agent.py:40-47 - No edge-case tests for impute_dataframe. The test suite...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:40-47

No edge-case tests for `impute_dataframe`. The test suite only covers a small, clean dataset with a few NaNs. Consider adding tests for: (1) an empty DataFrame, (2) a column where all values are NaN, (3) a DataFrame with no missing values, (4) a DataFrame missing the specified `time_col`, and (5) non-numeric data columns — to validate that the agent code handles these boundary cases gracefully rather than crashing.

```python
  def test_impute_dataframe_zero_config(self):
      # Test zero-config impute_dataframe
      infilled_df, diagnostics = impute_dataframe(
          self.df,
          time_col="timestamp",
          log_path=self.test_log,
          history_dir=self.test_history
      )
```

\+ 

* ```python
  def test_impute_dataframe_empty(self):
  ```
* ```python
      """Edge case: empty DataFrame should raise or return gracefully."""
  ```
* ```python
      empty_df = pd.DataFrame(columns=["timestamp", "signal"])
  ```
* ```python
      # Depending on intended behavior, expect either a meaningful error
  ```
* ```python
      # or an empty result — current test gap means this path is untested.
  ```
* ```python
      with self.assertRaises(ValueError):
  ```
* ```python
          impute_dataframe(empty_df, time_col="timestamp")
  ```
* 
* ```python
  def test_impute_dataframe_all_nan(self):
  ```
* ```python
      """Edge case: column with all NaN values."""
  ```
* ```python
      all_nan_df = self.df.copy()
  ```
* ```python
      all_nan_df["signal"] = np.nan
  ```
* ```python
      # Verify behavior — may raise, return all-NaN, or interpolate.
  ```
* ```python
      # Currently untested.
  ```
* 
* ```python
  def test_impute_dataframe_no_nans(self):
  ```
* ```python
      """Edge case: DataFrame with no missing values."""
  ```
* ```python
      clean_df = pd.DataFrame({"timestamp": [1, 2, 3], "signal": [10, 20, 30]})
  ```
* ```python
      result_df, _ = impute_dataframe(clean_df, time_col="timestamp")
  ```
* ```python
      pd.testing.assert_frame_equal(clean_df, result_df)
  ```
* 
* ```python
  def test_impute_dataframe_missing_time_col(self):
  ```
* ```python
      """Edge case: specified time column does not exist."""
  ```
* ```python
      with self.assertRaises((KeyError, ValueError)):
  ```
* ```python
          impute_dataframe(self.df, time_col="nonexistent")
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>5</id>
    <title>test_agent.py:131-133 - The test_agent_plot_diagnostics test does not clean up ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:131-133

The `test_agent_plot_diagnostics` test does not clean up the plot image in a `finally` block or `tearDown`. If `plot_diagnostics` raises an exception before the `os.remove` call on line 114, the temporary file `test_diagnostics_plot.png` will be left on disk, polluting the working directory.

* ```python
      try:
      # Verify the plot image was successfully created on disk
      self.assertTrue(os.path.exists(save_img))
  ```
* ```python
      finally:
  ```
* ```python
          if os.path.exists(save_img):
      os.remove(save_img)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>6</id>
    <title>setup.py:17-20 - Rigid platform detection breaks on non-Linux/macOS/Windows systems</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:17-20

**Rigid platform detection breaks on non-Linux/macOS/Windows systems** — The `else` branch assumes GCC with `-fopenmp`. On BSD, Solaris, Cygwin, or other Unix-like systems, this may cause hard compilation failures (e.g., Clang on FreeBSD uses `-fopenmp` only with libomp installed, Solaris cc uses `-xopenmp`). Consider checking the compiler (e.g., via `sysconfig.get_config_var('CC')`) or using a try-compile approach to detect actual OpenMP support instead of relying solely on `sys.platform`. At minimum, wrap in a try/except or add a fallback path.

  else:

* ```python
  # Linux (GCC)
  ```
* ```python
  # Assume GCC-compatible (Linux, etc.)
  ```
* ```python
  # NOTE: On BSD or other Unix-like systems, you may need different flags.
  ```
* ```python
  # Set NUIFI_NO_OPENMP=1 in environment to disable OpenMP explicitly.
  ```
* ```python
  if os.environ.get("NUIFI_NO_OPENMP"):
  ```
* ```python
      ext_compiler_args = []
  ```
* ```python
      ext_linker_args = []
  ```
* ```python
  else:
  ext_compiler_args = ["-fopenmp"]
  ext_linker_args = ["-fopenmp"]
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>7</id>
    <title>setup.py:19-20 - No compiler support check for OpenMP flags</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:19-20

**No compiler support check for OpenMP flags** — The `-fopenmp` flag is set unconditionally on Linux without verifying whether the compiler supports it. If the compiler lacks OpenMP support (e.g., a minimal Clang without libomp, or an older GCC), the build will fail with a cryptic error. Consider using a try-compile or checking if the flag is accepted (e.g., via `distutils.ccompiler` or a small subprocess test) and falling back gracefully with a warning.

  ext_compiler_args = ["-fopenmp"]
      ext_linker_args = ["-fopenmp"]

* ```python
  # Consider adding a try-compile check here:
  ```
* ```python
  # from distutils.ccompiler import new_compiler
  ```
* ```python
  # cc = new_compiler()
  ```
* ```python
  # if not cc.has_function('omp_get_num_threads', libraries=['gomp']):
  ```
* ```python
  #     print("WARNING: OpenMP not supported by compiler, disabling.")
  ```
* ```python
  #     ext_compiler_args = []
  ```
* ```python
  #     ext_linker_args = []
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>8</id>
    <title>setup.py:3-5 - setup.py lacks setup_requires for build-time dependencies</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:3-5

**`setup.py` lacks `setup_requires` for build-time dependencies** — `Cython` and `numpy` are imported at module level (lines 4-5), which means running `python setup.py build` without those packages installed will result in an `ImportError` with no guidance. While `pyproject.toml` declares these in `build-system.requires` for PEP 517 builds, direct `setup.py` invocations (common in older workflows) will crash. Consider adding a graceful error message, e.g., wrapping the imports in a try/except that tells users to install Cython and numpy first.

  from setuptools import setup, Extension

\+ 

* try:
from Cython.Build import cythonize
* except ImportError:
* ```python
  raise ImportError(
  ```
* ```python
      "Cython is required to build this package. "
  ```
* ```python
      "Install it via: pip install cython>=3.0.0"
  ```
* ```python
  )
  ```
* 
* try:
import numpy as np
* except ImportError:
* ```python
  raise ImportError(
  ```
* ```python
      "NumPy is required to build this package. "
  ```
* ```python
      "Install it via: pip install numpy>=1.20.0"
  ```
* ```python
  )
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>9</id>
    <title>benchmark.py:94-94 - Bug: ffill() and bfill() are not valid pandas methods.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:94-94

**Bug: `ffill()` and `bfill()` are not valid pandas methods.**

Pandas correct method names are `ffill()` (forward fill) and `bfill()` (back fill). Using `ffill()` and `bfill()` will raise `AttributeError` at runtime. Because this code is wrapped in a bare `except Exception` handler (line 103), the error will be silently swallowed and reported as `{"Error": str(e)}`, making this bug very hard to diagnose.

```python
      spline_infilled = spline_infilled.interpolate(method='linear', axis=0).ffill().bfill()
```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>10</id>
    <title>benchmark.py:48-53 - Data leakage: masked values at boundaries are replaced with ground truth.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:48-53

**Data leakage: masked values at boundaries are replaced with ground truth.**

The code replaces NaN values at the first and last indices with their corresponding ground truth values to "prevent interpolation extrapolation issues." However, this leaks information from `ground_truth` into the benchmark input data, invalidating the comparison. Spline and MICE methods will see the true values at boundaries and produce artificially better results. Instead, consider using forward/backward fill from the masked data itself, or use a masked-data-only imputation at boundaries.

* ```python
  # Ensure first and last values are not NaN to prevent interpolation extrapolation issues
  ```
* ```python
  # Ensure first and last values are not NaN using only masked data
  for c in range(n_channels):
  ```
* ```python
      if np.isnan(masked_data[0, c]):
  ```
* ```python
          masked_data[0, c] = ground_truth[0, c]
  ```
* ```python
      if np.isnan(masked_data[-1, c]):
  ```
* ```python
          masked_data[-1, c] = ground_truth[-1, c]
  ```
* ```python
      # Forward/backward fill from available data only
  ```
* ```python
      col = masked_data[:, c]
  ```
* ```python
      valid_idx = np.where(~np.isnan(col))[0]
  ```
* ```python
      if len(valid_idx) == 0:
  ```
* ```python
          continue  # can't fill, leave as NaN
  ```
* ```python
      if np.isnan(col[0]):
  ```
* ```python
          col[0] = col[valid_idx[0]]
  ```
* ```python
      if np.isnan(col[-1]):
  ```
* ```python
          col[-1] = col[valid_idx[-1]]
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>11</id>
    <title>benchmark.py:72-85 - Overly broad exception handling hides errors and makes debugging difficult.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:72-85

**Overly broad exception handling hides errors and makes debugging difficult.**

All four benchmark methods use bare `except Exception as e`, which catches everything including `AttributeError`, `TypeError`, `ImportError`, `KeyboardInterrupt`, etc. This can silently swallow real bugs (e.g., the `ffill()`/`bfill()` typo on line 98) and mask configuration or dependency problems. Consider catching only expected exceptions (`ValueError`, `RuntimeError`, etc.) and letting unexpected errors propagate so they can be diagnosed.

```python
  try:
      nufi_infilled = nufi.fit_transform(df_masked, timestamps=timestamps)
      nufi_time = time.time() - start
      
      nufi_rmse = np.sqrt(np.mean((df_truth.to_numpy() - nufi_infilled.to_numpy()) ** 2))
      nufi_cov_err = np.linalg.norm(true_cov - nufi_infilled.cov().to_numpy(), ord='fro')
      
      results["NUFI"] = {
          "RMSE": float(nufi_rmse),
          "Covariance Error (Frobenius)": float(nufi_cov_err),
          "Runtime (s)": float(nufi_time)
      }
```

* ```python
  except Exception as e:
  ```
* ```python
  except (ValueError, RuntimeError, ImportError) as e:
      results["NUFI"] = {"Error": str(e)}
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>12</id>
    <title>benchmark.py:71-71 - NUFI reproducibility: no random seed mechanism available.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:71-71

**NUFI reproducibility: no random seed mechanism available.**

`NufiImputer` is instantiated without a random seed, and the class does not expose a `random_state` parameter. If any internal operations rely on randomness (e.g., stochastic multiple imputation, GCV candidate shuffling, or PyTorch operations), benchmark results will not be reproducible across runs. Consider documenting this limitation or adding a `random_state` parameter to `NufiImputer` for deterministic benchmarks.

```python
  nufi = NufiImputer(device='cpu', covariance_compensation=True, n_frequencies='auto', alpha='auto')
```

* ```python
  # NOTE: NufiImputer does not expose a random_state parameter.
  ```
* ```python
  # Benchmark reproducibility depends on the imputer being deterministic.
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>13</id>
    <title>wrappers.py:57-62 - Missing validation of fit_transform output row count in both single-index and MultiIndex paths.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:57-62

**Missing validation of `fit_transform` output row count in both single-index and MultiIndex paths.**  
The `infill_dataframe` function (line 22) and the `imputer_apply` inner function (line 53) both call `imputer.fit_transform(...)` and either return the result directly (single-index) or blindly assign `group_sorted.index` (MultiIndex). If `fit_transform` drops or adds rows, this will either cause a `ValueError` for length mismatch or silently corrupt data. Add a row-count assertion after `fit_transform` to ensure the output shape matches the input.

```python
      temp_df = pd.DataFrame(group_sorted.to_numpy(), index=timestamps, columns=group_sorted.columns)
      infilled_temp = imputer.fit_transform(temp_df)
```

* 
* ```python
      if len(infilled_temp) != len(group_sorted):
  ```
* ```python
          raise ValueError(
  ```
* ```python
              f"Imputer returned {len(infilled_temp)} rows, "
  ```
* ```python
              f"expected {len(group_sorted)}. "
  ```
* ```python
              "The NufiImputer must preserve row count."
  ```
* ```python
          )
      
      # Restore MultiIndex structure
      infilled_temp.index = group_sorted.index
      return infilled_temp
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>14</id>
    <title>wrappers.py:21-23 - High: time_col is silently removed from the feature set.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:21-23

**High: `time_col` is silently removed from the feature set.**

When `time_col is not None`, `set_index(time_col)` moves that column to the index and drops it from the DataFrame columns. This means the time column is excluded from the imputation model, even though it often carries valuable temporal signal (e.g., seasonality, trend). Users may not expect this side effect. In the MultiIndex path, timestamps are extracted and passed as the index only, not as a feature column, which is a similar concern.

**Suggestion:** Either document this behaviour prominently, or offer an option to keep the time column as a feature (e.g., by copying it before setting the index). Consider whether the imputer should receive time as both index and feature.

```python
  # If a specific column is defined as time, set it as index
  if time_col is not None:
```

* ```python
      # Preserve time column as a feature for the imputer
  ```
* ```python
      time_values = pd_df[time_col].copy()
      pd_df = pd_df.set_index(time_col)
  ```
* ```python
      pd_df[time_col] = time_values  # keep as feature
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>15</id>
    <title>wrappers.py:51-53 - Medium: Groups are silently sorted by time level, which may surprise callers.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:51-53

**Medium: Groups are silently sorted by time level, which may surprise callers.**

`group.sort_index(level=time_level)` reorders rows within each entity group without any warning or documentation. If the caller expects the original row order to be preserved in non‑time columns or in the output, this silent reordering can lead to subtle bugs downstream.

**Suggestion:** Document this behaviour clearly in the docstring, and consider adding a `sort` parameter (default `True`) so callers can opt out when they are confident the data is already chronologically ordered.

```python
  def imputer_apply(group):
```

* ```python
      # We need to sort index by time level to ensure proper chronological order
  ```
* ```python
      # Sort index by time level to ensure proper chronological order.
  ```
* ```python
      # NOTE: This reorders the group's rows; the output index order will
  ```
* ```python
      # reflect the sorted order, not the original input order.
      group_sorted = group.sort_index(level=time_level)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>16</id>
    <title>wrappers.py:57-57 - Medium: Column names are lost when creating the temporary DataFrame.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:57-57

**Medium: Column names are lost when creating the temporary DataFrame.**

`group_sorted.to_numpy()` strips column names; they are re‑supplied from `group_sorted.columns`. However, if `fit_transform` internally relies on column names (e.g., for feature‑specific imputation strategies), this should work. The bigger issue is that `to_numpy()` may produce a copy of the data and, more importantly, it coerces all columns to a common dtype (object if heterogeneous). This can change the data representation seen by the imputer compared to the original DataFrame, potentially affecting imputation quality.

**Suggestion:** Use `group_sorted.reset_index(drop=True)` or pass `group_sorted` directly (with a simplified index) to preserve dtypes and avoid unnecessary copies. For example:

```python
temp_df = group_sorted.droplevel(entity_level)
```

* ```python
      temp_df = pd.DataFrame(group_sorted.to_numpy(), index=timestamps, columns=group_sorted.columns)
  ```
* ```python
      # Avoid to_numpy() which coerces dtypes; copy with a clean index instead
  ```
* ```python
      temp_df = group_sorted.copy()
  ```
* ```python
      temp_df.index = timestamps
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>17</id>
    <title>wrappers.py:10-16 - Low: Fragile cuDF detection relying on __name__ and __module__ introspection.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:10-16

**Low: Fragile cuDF detection relying on `__name__` and `__module__` introspection.**

The current check (`type(df).__name__ == "DataFrame"` and `type(df).__module__.startswith("cudf")`) is brittle: it would fail on subclasses, renamed modules, or if cuDF ever changes its internal module layout. Additionally, the fallback `df.copy()` on line 14 is an eager copy even when `set_index` would already produce a new DataFrame.

**Suggestion:** Use a try/except import pattern:

```python
try:
    import cudf
    is_cudf = isinstance(df, cudf.DataFrame)
except ImportError:
    is_cudf = False
```

This is more robust and more readable.

* ```python
  try:
  ```
* ```python
      import cudf
  ```
* ```python
      is_cudf = isinstance(df, cudf.DataFrame)
  ```
* ```python
  except ImportError:
  is_cudf = False
  ```
* ```python
  if type(df).__name__ == "DataFrame" and type(df).__module__.startswith("cudf"):
  ```
* ```python
      is_cudf = True
  ```
* ```python
      # Convert cuDF to Pandas for sklearn compatibility
  ```

\+ 

* ```python
  if is_cudf:
      pd_df = df.to_pandas()
  else:
      pd_df = df.copy()
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>18</id>
    <title>test_imputer.py:5-5 - Dead import: infill_dataframe is imported but never exercised in any test.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:5-5

**Dead import: `infill_dataframe` is imported but never exercised in any test.**

This is both a maintainability issue (unused import adds noise) and a test coverage gap — `infill_dataframe` is a public API function with zero test coverage. Either add a dedicated test for it or remove the import if it's intentionally not tested here.

* from nufi.wrappers import infill_dataframe, infill_multiindex_dataframe
* from nufi.wrappers import infill_multiindex_dataframe
]]></description>
  </task>
  <task status="COMPLETED">
    <id>19</id>
    <title>test_imputer.py:103-112 - Test coverage gap: single-column input only.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:103-112

**Test coverage gap: single-column input only.**

Stochastic imputation with multiple columns should also verify that the covariance structure across columns is respected (e.g., imputed values in column A and column B should maintain a realistic correlation). A single-column test cannot detect cross-column correlation errors that may arise in multi-column stochastic imputation.

  def test_stochastic_imputation():

* ```python
  # Verify that stochastic multiple imputation produces non-deterministic filled values
  ```
* ```python
  # on missing spots while preserving non-nan values.
  ```
* ```python
  # Single-column case
  X = np.array([
      [1.0],
      [np.nan],
      [3.0],
      [np.nan],
      [5.0]
  ], dtype=np.float64)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>20</id>
    <title>test_imputer.py:66-78 - Insufficient validation: direct vs CG solver consistency is not checked.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:66-78

**Insufficient validation: direct vs CG solver consistency is not checked.**

The test only verifies that both solvers produce non-NaN output and preserve non-NaN values, but it never asserts that the two solvers produce *similar* imputations for the missing values. Without a tolerance-based comparison (e.g., `np.allclose(X_direct, X_cg, atol=1e-6)`), the test cannot detect solver divergence, numerical instability, or incorrect CG convergence.

```python
  imputer_direct = NufiImputer(method='direct', solver='direct', alpha=1e-3, covariance_compensation=False)
  X_direct = imputer_direct.fit_transform(X)
  
  imputer_cg = NufiImputer(method='direct', solver='cg', alpha=1e-3, covariance_compensation=False)
  X_cg = imputer_cg.fit_transform(X)
  
  assert not np.any(np.isnan(X_direct))
  assert not np.any(np.isnan(X_cg))
  assert X_direct.shape == X.shape
  assert X_cg.shape == X.shape
  # Check that non-nan values are preserved in both cases
  assert X_direct[0, 0] == 2.0
  assert X_cg[0, 0] == 2.0
```

* ```python
  # Verify direct and CG solvers produce consistent imputations
  ```
* ```python
  assert np.allclose(X_direct, X_cg, atol=1e-6)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>21</id>
    <title>test_imputer.py:92-101 - Insufficient validation: GCV test only checks parameter existence, not imputation quality.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:92-101

**Insufficient validation: GCV test only checks parameter existence, not imputation quality.**

The test verifies that alpha and n_frequencies are positive and present, but it never validates that the resulting imputation is reasonable (e.g., imputed values fall within a plausible range relative to observed data). Consider adding assertions that imputed values are within reasonable bounds (e.g., within [min_observed, max_observed] or at least not extreme outliers).
]]></description>
  </task>
  <task status="COMPLETED">
    <id>22</id>
    <title>test_imputer.py:7-24 - Missing edge-case tests across the entire file.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:7-24

**Missing edge-case tests across the entire file.**

The test suite lacks coverage for critical boundary conditions:

* All-NaN column/row (should this raise an error or produce a sensible default?)
* Single-row input
* Single-column input with all values observed (no NaNs — no-op path)
* Very large input that may trigger numerical instability
* Invalid parameters (e.g., negative alpha, invalid method string, `n_frequencies` > number of observations)
* `fit` / `transform` called out of order (calling `transform` without `fit` first)

Adding these would significantly improve confidence in the imputer's robustness.
]]></description>
  </task>
  <task status="COMPLETED">
    <id>23</id>
    <title>test_covariance.py:22-25 - Overly lenient thresholds in test_covariance.py reduce test effectiveness.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:22-25

**Overly lenient thresholds in `test_covariance.py` reduce test effectiveness.**  
The derivative threshold test permits maximum first derivative up to 1.0 and second derivative up to 0.5, which are 10×–50× above expected values for sin(t), making it pass trivially even for linear interpolation. The covariance tolerance (`rtol=1e-1, atol=1e-1`) allows up to 10% relative error plus 0.1 absolute error per entry, masking ~20% errors. Tighten these thresholds to more realistic bounds (e.g., <0.2 for dx, <0.02 for ddx; and rtol=1e-2, atol=1e-2) to ensure the tests actually validate Fourier-specific improvements.

* ```python
  # Check that there are no extreme, discontinuous jumps in the first & second derivatives
  ```
* ```python
  # Linear interpolation would show a sharp jump at the boundary, but Fourier is completely smooth
  ```
* ```python
  assert np.max(np.abs(dx)) < 1.0  # smooth derivative bounds
  ```
* ```python
  assert np.max(np.abs(ddx)) < 0.5 # smooth second derivative bounds
  ```
* ```python
  # sin(t) has max derivative 1.0; with dt ≈ 0.101, max |dx| ≈ 0.101, max |ddx| ≈ 0.0102
  ```
* ```python
  assert np.max(np.abs(dx)) < 0.2   # tight bound: ~2× the expected max
  ```
* ```python
  assert np.max(np.abs(ddx)) < 0.02  # tight bound: ~2× the expected max
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>24</id>
    <title>test_covariance.py:22-25 - Missing baseline comparison: Neither test compares the Fo...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:22-25

Missing baseline comparison: Neither test compares the Fourier-based imputer against a trivial baseline (e.g., linear or spline interpolation). The library’s value proposition is that it outperforms naive methods on smoothness and covariance preservation, but these tests only assert absolute properties without demonstrating superiority. Consider adding a second imputation pass with linear interpolation and asserting that the Fourier result has *lower* derivative spikes / *smaller* covariance deviation than the baseline. Without this contrast, the tests cannot validate the core motivation of the library.

* ```python
  # Check that there are no extreme, discontinuous jumps in the first & second derivatives
  ```
* ```python
  # Linear interpolation would show a sharp jump at the boundary, but Fourier is completely smooth
  ```
* ```python
  assert np.max(np.abs(dx)) < 1.0  # smooth derivative bounds
  ```
* ```python
  assert np.max(np.abs(ddx)) < 0.5 # smooth second derivative bounds
  ```
* ```python
  # Baseline: linear interpolation for comparison
  ```
* ```python
  from scipy.interpolate import interp1d
  ```
* ```python
  valid = ~np.isnan(signal)
  ```
* ```python
  linear_fill = interp1d(t[valid], signal[valid], kind='linear', fill_value='extrapolate')(t)
  ```
* ```python
  lin_dx = np.max(np.abs(np.diff(linear_fill)))
  ```
* ```python
  lin_ddx = np.max(np.abs(np.diff(np.diff(linear_fill))))
  ```
* 
* ```python
  # Fourier must be at least as smooth (lower or equal max derivative spikes)
  ```
* ```python
  assert np.max(np.abs(dx)) <= lin_dx * 1.01
  ```
* ```python
  assert np.max(np.abs(ddx)) < 0.02
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>25</id>
    <title>impute.py:183-209 - Bug: Stochastic noise double‑scaling under covariance compensation.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:183-209

**Bug: Stochastic noise double‑scaling under covariance compensation.**

When `covariance_compensation` is active, `reconstructed_np` is multiplied by `cov_scale` at line 195, but the residual for the stochastic standard deviation is then computed as `X_data[obs_mask, col_idx] - reconstructed_np[obs_mask]` at line 215 — i.e., original (unscaled) observations minus already‑scaled reconstruction. This mixes two different scales, inflating the residual standard deviation. Later at line 228, the noise samples are multiplied by `cov_scale` *again*, resulting in a net `cov_scale²` inflation of imputation uncertainty.

**Suggested fix:** Either (a) compute the residual *before* the covariance‑compensation scaling of `reconstructed_np`, or (b) scale `X_data[obs_mask]` by `cov_scale` when computing the residual so that both terms live in the same space. Apply `cov_scale` to the noise only once (in the final `noise * cov_scale` step).

* ```python
          # Store pre-compensation reconstruction for proper residual computation
  ```
* ```python
          reconstructed_raw = reconstructed_np.copy()
  ```
* ```python
          # If covariance compensation is computed, align the reconstructed scale
  ```
* ```python
          cov_scale = 1.0
          if self.covariance_compensation and self.d_ is not None:
              cov_scale = np.sqrt(np.abs(np.diag(self.d_)[col_idx]))
              if cov_scale > 0:
                  reconstructed_np = reconstructed_np * cov_scale
          
          # Fill only the NaNs
          nan_mask = np.isnan(X_data[:, col_idx])
          if np.any(nan_mask):
              if stochastic:
                  # Compute residual standard deviation on observed values
                  obs_mask = ~nan_mask
                  if np.any(obs_mask):
  ```
* ```python
                      residual = X_data[obs_mask, col_idx] - reconstructed_np[obs_mask]
  ```
* ```python
                      # Use unscaled reconstruction so both terms are in the same space
  ```
* ```python
                      residual = X_data[obs_mask, col_idx] - reconstructed_raw[obs_mask]
                      residual_std = np.std(residual) if len(residual) > 1 else 0.1
                      if np.isnan(residual_std) or residual_std == 0:
                          residual_std = 0.1
                  else:
                      residual_std = 0.1
                      
                  # Generate noise from posterior process scaled by uncertainty parameters
                  noise = np.random.normal(0, stochastic_scale * residual_std, size=nan_mask.sum())
  ```
* ```python
                  if self.covariance_compensation and self.d_ is not None:
  ```
* ```python
                      cov_scale = np.sqrt(np.abs(np.diag(self.d_)[col_idx]))
  ```
* ```python
                      if cov_scale > 0:
  ```
* ```python
                  if self.covariance_compensation and self.d_ is not None and cov_scale > 0:
                          noise = noise * cov_scale
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>26</id>
    <title>impute.py:117-120 - Bug: All‑NaN columns silently included in covariance decomposition.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:117-120

**Bug: All‑NaN columns silently included in covariance decomposition.**

In `fit()`, every column — even those with `N_val == 0` (entirely NaN) — is added to `X_list` and passed to `covariance_compensation`. While `compute_ND_NUDFT` returns zero‑vectors for such columns, the resulting covariance matrix will contain rows/columns of zeros, making it rank‑deficient. `scipy.linalg.ldl` may fail outright or return a numerically unstable decomposition that corrupts the downstream `d_` matrix used in `transform()` for scaling.

**Suggested fix:** Skip columns with no valid data when building `X_list`, or at minimum issue a warning. The `d_` diagonal can be padded with 1.0 (identity scale) for those columns so that `transform()` indexing stays aligned.

```python
      # Build X_list of DatasetObj for covariance computation
```

* ```python
      # Skip columns with no valid data to avoid degenerate covariance
      X_list = []
  ```
* ```python
      valid_col_mask = []
      for col_idx in range(X_data.shape[1]):
  ```
* ```python
          X_list.append(DatasetObj(self.timestamps_, X_data[:, col_idx]))
  ```
* ```python
          col_data = X_data[:, col_idx]
  ```
* ```python
          if np.any(~np.isnan(col_data) & ~np.isnan(self.timestamps_)):
  ```
* ```python
              X_list.append(DatasetObj(self.timestamps_, col_data))
  ```
* ```python
              valid_col_mask.append(True)
  ```
* ```python
          else:
  ```
* ```python
              valid_col_mask.append(False)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>27</id>
    <title>impute.py:102-103 - Reliability: torch.linalg.svd has no error handling in the GCV loop.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:102-103

**Reliability: `torch.linalg.svd` has no error handling in the GCV loop.**

SVD at line 97 is called for every candidate `n_f` of every column inside `fit()`. If `A` is ill‑conditioned (e.g., near‑duplicate timestamps, extreme frequency spacing) or the matrix is too large for GPU memory, `torch.linalg.svd` will raise a `RuntimeError` that crashes the entire fit with no recovery path. The same risk exists in `transform()` when `solve_tikhonov_nudft` uses the direct solver (`torch.linalg.solve`).

**Suggested fix:** Wrap the SVD (and the direct solve path) in try/except, log a warning, and fall back to a safe default (e.g., the previous `best_n_freq` / `best_alpha` for that column, or the CG solver).

```python
              # Compute GCV score
```

* ```python
            try:
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
  ```
* ```python
            except RuntimeError as e:
  ```
* ```python
                # SVD failed (ill-conditioned or OOM); keep current best
  ```
* ```python
                import warnings
  ```
* ```python
                warnings.warn(f"SVD failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
  ```
* ```python
                continue
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>28</id>
    <title>impute.py:96-103 - Performance: Redundant SVD computation when alpha='auto'.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:96-103

**Performance: Redundant SVD computation when `alpha='auto'`.**

When `self.alpha == 'auto'`, `optimize_alpha_gcv(A, t_data, candidate_alphas)` (line 93) computes an SVD internally. Immediately afterward, line 97 computes *another* SVD on the exact same matrix `A` to evaluate the GCV score. Since SVD is the most expensive step in the GCV loop, this doubles the cost per candidate.

**Suggested fix:** Refactor `optimize_alpha_gcv` to also return the SVD components `(U, S, y_tilde, y_norm_sq)` or its best GCV score, so the caller can reuse them instead of recomputing.

```python
              if self.alpha == 'auto':
                  candidate_alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
```

* ```python
                opt_alpha = optimize_alpha_gcv(A, t_data, candidate_alphas)
  ```
* ```python
                # optimize_alpha_gcv internally computes SVD; reuse its results
  ```
* ```python
                from nufi.kernels.torch_kernels import optimize_alpha_gcv
  ```
* ```python
                opt_alpha, U, S, y_tilde, y_norm_sq = optimize_alpha_gcv(
  ```
* ```python
                    A, t_data, candidate_alphas, return_svd=True)
            else:
                opt_alpha = self.alpha if self.alpha is not None else 1e-4
  ```
* 
* ```python
            # Compute GCV score
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
  ```
* ```python
                y_complex = t_data.to(torch.complex128)
  ```
* ```python
                y_tilde = torch.matmul(U.adjoint(), y_complex)
  ```
* ```python
                y_norm_sq = torch.sum(torch.abs(y_complex) ** 2)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>29</id>
    <title>agent.py:238-241 - Critical: Missing import torch causes runtime NameError.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:238-241

**Critical: Missing `import torch` causes runtime `NameError`.**  
The diagnostics loop uses `torch.tensor`, `torch.float64`, `torch.real`, `torch.sum`, and `torch.exp` (lines 238-241), but the module only imports `get_device` and `solve_tikhonov_nudft` from `nufi.kernels.torch_kernels` — `torch` itself is never imported. For any column with valid observations, this will crash with `NameError: name 'torch' is not defined`.

Fix: Add `import torch` at the top of the file.

```python
      t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
      t_times = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
      exponent = 2.0j * np.pi * t_times.unsqueeze(1) * t_f_k.unsqueeze(0)
      reconstructed = torch.real(torch.sum(F.unsqueeze(0) * torch.exp(exponent), dim=1))
```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>30</id>
    <title>agent.py:27-33 - Thread-safety: log_transformation and save_snapshot write to a shared log file without any locking.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:27-33

**Thread-safety: `log_transformation` and `save_snapshot` write to a shared log file without any locking.**  
If `impute_dataframe` is called concurrently (e.g., from multiple threads, async tasks, or parallel DAG steps), interleaved `write()` calls will produce corrupted JSON lines. `save_snapshot` also risks TOCTOU races between `list_versions` and file writes.

Fix: Use `threading.Lock` (or a file-based lock like `fcntl.flock` / `portalocker` for multi-process safety) to serialize access to the log file and history directory.

```python
  def log_transformation(self, log_entry: dict):
      """Appends a transformation log entry to the log file in append-only mode."""
      try:
```

* ```python
        with self._lock:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        raise TransformationLoggingError(f"Failed to write to transformation log: {e}")
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>31</id>
    <title>agent.py:37-37 - Version ID collision risk from millisecond-precision timestamps.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:37-37

**Version ID collision risk from millisecond-precision timestamps.**  
`int(time.time() * 1000)` produces IDs like `ver_1751304061000`. In high-frequency or near-simultaneous calls (same millisecond), duplicate IDs are generated, causing silent snapshot overwrites and permanent data lineage loss.

Fix: Use `uuid.uuid4().hex[:12]` or a monotonic counter in addition to the timestamp to guarantee uniqueness.

* ```python
    version_id = f"ver_{int(time.time() * 1000)}"
  ```
* ```python
    version_id = f"ver_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>32</id>
    <title>agent.py:19-25 - Path traversal vulnerability: log_path and history_dir are used directly in file I/O without sanitization.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:19-25

**Path traversal vulnerability: `log_path` and `history_dir` are used directly in file I/O without sanitization.**  
If `impute_dataframe` is exposed to user-controlled input (e.g., a web API or CLI), an attacker can supply paths like `../../etc/cron.d/evil` to write to arbitrary locations on the filesystem via `os.makedirs` and file `open()`.

Fix: Validate that resolved paths stay within an allowed base directory using `os.path.realpath` and a prefix check, or reject paths containing `..` segments.

```python
  def __init__(self, log_path: str = "nufi_transformations.log", history_dir: str = ".nufi_history"):
```

* ```python
    self.log_path = log_path
  ```
* ```python
    self.history_dir = history_dir
  ```
* ```python
    self.log_path = os.path.realpath(log_path)
  ```
* ```python
    self.history_dir = os.path.realpath(history_dir)
  ```
* ```python
    # Optional: assert self.history_dir.startswith(allowed_base)
    try:
        os.makedirs(self.history_dir, exist_ok=True)
    except Exception as e:
        raise TransformationLoggingError(f"Failed to create history directory: {e}")
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>33</id>
    <title>agent.py:178-178 - Unvalidated float64 conversion of DataFrame index causes cryptic crashes in both impute_dataframe and plot_diagnostics.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:178-178

**Unvalidated `float64` conversion of DataFrame index causes cryptic crashes in both `impute_dataframe` and `plot_diagnostics`.**  
Both functions call `df_copy.index.to_numpy(dtype=np.float64)` without checking if the index is numeric. If the index is datetime, string, or other non-numeric type, this raises a `ValueError` with an unhelpful error message. Add a validation step (e.g., `pd.api.types.is_numeric_dtype`) and raise a clear `TypeError` explaining that a numeric index or `time_col` is required.

* ```python
  if not pd.api.types.is_numeric_dtype(df_copy.index):
  ```
* ```python
    raise TypeError(
  ```
* ```python
        f"DataFrame index must be numeric (timestamps). "
  ```
* ```python
        f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col`."
    )
  timestamps = df_copy.index.to_numpy(dtype=np.float64)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>34</id>
    <title>agent.py:377-380 - plot_diagnostics hardcodes solver='direct', ignoring the user's original solver choice.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:377-380

**`plot_diagnostics` hardcodes `solver='direct'`, ignoring the user's original solver choice.**  
The PSD recomputation in `plot_diagnostics` always uses the direct solver, even if the user originally selected `solver='cg'`. This can silently produce different (potentially inconsistent) spectral estimates than those used during infilling, and may be much slower for large problems where CG was intentionally chosen.

Fix: Accept and forward the `solver` (and `max_iter`, `tol`, `device`) parameters to `plot_diagnostics`, or extract them from the `diagnostics` dict.

```python
          F = solve_tikhonov_nudft(
              v_timestamps, v_data, f_k, opt_alpha,
```

* ```python
            solver='direct', max_iter=100, tol=1e-5, device=None
  ```
* ```python
            solver=solver, max_iter=max_iter, tol=tol, device=device
        )
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>35</id>
    <title>torch_kernels.py:43-61 - HIGH: Incorrect NUDFT formula — computes wrong spectrum.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:43-61

**HIGH: Incorrect NUDFT formula — computes wrong spectrum.** The code uses time differences `p_n = np.diff(v_timestamps)` in the exponent instead of absolute timestamps, then applies `v_data[:-1]` (discarding the last valid data point). The exponent `-2πi * p_n[k] * f_k[k]` is an element-wise product of a difference and a specific frequency index, which is not the standard NUDFT sum `Σ x_n · exp(−2πi · t_n · f_k)`. Additionally, only the first `len(p_n)` output bins are filled; the remaining `N - len(p_n)` bins are left as zero. This produces a fundamentally wrong spectrum.

* ```python
    p_n = np.diff(v_timestamps)
  ```
* ```python
    # Avoid division by zero if all timestamps are identical
  ```
* ```python
    min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
  ```
* ```python
    max_sampling_rate = 1.0 / min_p
  ```
* ```python
    nyquist_frequency = max_sampling_rate / 2.0
  ```
* ```python
    N = len(data)
  ```
* ```python
    f_k = np.linspace(0, nyquist_frequency, N)
  ```
* ```python
    # Use absolute timestamps, not differences
  ```
* ```python
    t_timestamps = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
  ```
* ```python
    t_data_all = torch.tensor(v_data, dtype=torch.float64, device=dev)
  ```
* ```python
    # Move to PyTorch
  ```
* ```python
    t_p_n = torch.tensor(p_n, dtype=torch.float64, device=dev)
  ```
* ```python
    t_f_k = torch.tensor(f_k[:len(p_n)], dtype=torch.float64, device=dev)
  ```
* ```python
    t_data = torch.tensor(v_data[:-1], dtype=torch.float64, device=dev)
  ```
* ```python
    # Estimate Nyquist frequency from median sampling interval
  ```
* ```python
    if len(v_timestamps) > 1:
  ```
* ```python
        min_p = np.nanmin(np.diff(v_timestamps))
  ```
* ```python
        nyquist_frequency = 0.5 / max(min_p, 1e-12)
  ```
* ```python
    else:
  ```
* ```python
        nyquist_frequency = 1.0
  ```
* ```python
    # Vectorized exponent computation
  ```
* ```python
    # exponent = -2j * pi * p_n * f_k
  ```
* ```python
    exponent = -2.0j * np.pi * t_p_n * t_f_k
  ```
* ```python
    summation = torch.zeros(N, dtype=torch.complex128, device=dev)
  ```
* ```python
    summation[:len(t_data)] = t_data.to(torch.complex128) * torch.exp(exponent)
  ```
* ```python
    f_k = torch.linspace(0, nyquist_frequency, N, dtype=torch.float64, device=dev)
  ```
* 
* ```python
    # Standard NUDFT: A[n,k] = exp(-2πi * t_n * f_k), then sum over n
  ```
* ```python
    exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
  ```
* ```python
    summation = torch.sum(t_data_all.to(torch.complex128).unsqueeze(1) * torch.exp(exponent), dim=0)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>36</id>
    <title>torch_kernels.py:117-117 - HIGH: Imaginary part discarded in covariance calculation — loses phase information.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:117-117

**HIGH: Imaginary part discarded in covariance calculation — loses phase information.** The line `tensor.cpu().numpy().real` discards the imaginary components of the complex NUDFT result before computing the covariance matrix. For complex-valued frequency-domain signals, the imaginary part encodes phase relationships that are essential for correct covariance estimation. Using only the real part will produce a distorted covariance matrix, leading to incorrect LDL decomposition and downstream imputation errors.

* ```python
    flat_data.append(tensor.cpu().numpy().real) # Process real part for covariance
  ```
* ```python
    # Use the full complex magnitude or preserve both real and imaginary parts.
  ```
* ```python
    # Option A: use magnitude (if phase is irrelevant):
  ```
* ```python
    # flat_data.append(np.abs(tensor.cpu().numpy()))
  ```
* ```python
    # Option B: stack real and imaginary as separate features:
  ```
* ```python
    arr = tensor.cpu().numpy()
  ```
* ```python
    flat_data.append(np.concatenate([arr.real, arr.imag]))
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>37</id>
    <title>torch_kernels.py:90-91 - HIGH: Assumes sorted timestamps — produces wrong grid for non-monotonic data.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:90-91

**HIGH: Assumes sorted timestamps — produces wrong grid for non-monotonic data.** `timestamps[0]` and `timestamps[-1]` are used as the bounds of `np.linspace`, but if timestamps are not sorted, these are not the true min/max. The resulting uniform grid can be misaligned or even reversed, causing `np.interp` to produce incorrect values and giving a meaningless FFT.

* ```python
    # Generate uniform grid
  ```
* ```python
    uniform_grid = np.linspace(timestamps[0], timestamps[-1], N)
  ```
* ```python
    # Use actual min/max of valid timestamps
  ```
* ```python
    t_min, t_max = np.min(v_timestamps), np.max(v_timestamps)
  ```
* ```python
    uniform_grid = np.linspace(t_min, t_max, N)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>38</id>
    <title>torch_kernels.py:210-217 - HIGH: No validation of regularization parameter alpha — zero or negative values break the solver.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:210-217

**HIGH: No validation of regularization parameter `alpha` — zero or negative values break the solver.** `alpha` is passed directly into the linear system `A^H A + αI`. If `alpha <= 0`, the matrix may become singular or indefinite, causing `torch.linalg.solve` to crash or return garbage. Add a guard that raises `ValueError` for non-positive alpha.

* ```python
  if alpha <= 0:
  ```
* ```python
    raise ValueError(f"Regularization parameter alpha must be positive, got {alpha}")
  ```
* ```python
  dev = get_device(device)
  t_timestamps = torch.tensor(timestamps, dtype=torch.float64, device=dev)
  t_data = torch.tensor(data, dtype=torch.float64, device=dev)
  t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)

  # Build Fourier mapping matrix A: shape (N, M)
  # A_nk = exp(2*pi*i * f_k * t_n)
  exponent = 2.0j * np.pi * t_timestamps.unsqueeze(1) * t_f_k.unsqueeze(0)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>39</id>
    <title>torch_kernels.py:125-125 - MEDIUM: NaN entries in covariance matrix silently replaced by zero — masks data quality issues.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:125-125

**MEDIUM: NaN entries in covariance matrix silently replaced by zero — masks data quality issues.**  
`fillna(0.0)` replaces any NaN in the covariance matrix with 0.0. NaN values in a covariance matrix typically indicate degenerate data (e.g., zero-variance columns, all-NaN time series). Replacing them with zero hides these problems and can produce a misleading LDL decomposition that silently propagates errors through the imputation pipeline.

* ```python
    covariance_matrix = df.cov().fillna(0.0).to_numpy()
  ```
* ```python
    covariance_matrix = df.cov().to_numpy()
  ```
* ```python
    # Warn or raise if NaNs are present — they indicate degenerate input
  ```
* ```python
    if np.any(np.isnan(covariance_matrix)):
  ```
* ```python
    import warnings
  ```
* ```python
    warnings.warn("Covariance matrix contains NaN entries; degenerate columns detected.")
  ```
* ```python
    # Optionally fill diagonals with small positive values for numerical stability:
  ```
* ```python
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>40</id>
    <title>torch_kernels.py:59-59 - MEDIUM: Inconsistent sign convention between NUDFT and Tikhonov solver.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:59-59

**MEDIUM: Inconsistent sign convention between NUDFT and Tikhonov solver.** `compute_ND_NUDFT` uses `exp(−2πi · t · f)` while `solve_tikhonov_nudft` builds the design matrix as `exp(+2πi · t · f)`. If both are meant to use the same Fourier convention, this discrepancy will cause phase inversions and incorrect results when the two functions are used together in the same pipeline.

* ```python
    exponent = -2.0j * np.pi * t_p_n * t_f_k
  ```
* ```python
    # Standard NUDFT: X(f) = Σ x(t) · exp(−2πi · t · f)
  ```
* ```python
    exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
  ```
]]></description>
  </task>
  <task status="COMPLETED">
    <id>41</id>
    <title>pyproject.toml:20-21 - Manual package listing risks omissions</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:20-21

**Manual package listing risks omissions**: Explicitly listing `["nufi", "nufi.kernels"]` means any new subpackage (e.g., `nufi.utils`, `nufi.io`) will be silently excluded from installation until this list is manually updated. Use `setuptools`' auto-discovery directive instead: `packages = ["find:auto"]` (or `find:` with `[tool.setuptools.packages.find]` for more control). This prevents incomplete installs as the package grows.

  [tool.setuptools]

* packages = ["nufi", "nufi.kernels"]
* packages = ["find:auto"]
* 
* [tool.setuptools.packages.find]
* include = ["nufi*"]
]]></description>
  </task>
  <task status="COMPLETED">
    <id>42</id>
    <title>pyproject.toml:2-2 - Missing oldest-supported-numpy for ABI compatibility</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:2-2

**Missing `oldest-supported-numpy` for ABI compatibility**: The `setup.py` builds Cython extensions that link against numpy's C API (via `np.get_include()`), but the build dependency in `pyproject.toml` only pins `numpy>=1.20.0`. When building wheels, the build environment will install the *latest* numpy, and the resulting binary will be linked against that newer numpy ABI. Users running an older (but still satisfying `>=1.20.0`) numpy at runtime will encounter ABI mismatch errors. The standard fix is to add `oldest-supported-numpy` to build requirements, which installs the *oldest* numpy version compatible with the target Python, ensuring maximum ABI forward-compatibility. Replace `"numpy>=1.20.0"` with `"oldest-supported-numpy"` in `build-system.requires`.

* requires = ["setuptools>=61.0.0", "wheel", "Cython>=3.0.0", "numpy>=1.20.0"]
* requires = ["setuptools>=61.0.0", "wheel", "Cython>=3.0.0", "oldest-supported-numpy"]
]]></description>
  </task>
  <task status="COMPLETED">
    <id>43</id>
    <title>pyproject.toml:12-18 - Hard torch dependency may mislead GPU expectations</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:12-18

**Hard `torch` dependency may mislead GPU expectations**: The package description promises "GPU-accelerated" infilling, but `torch>=2.0.0` as a mandatory dependency installs the CPU-only PyTorch wheel by default (the CUDA-enabled wheels use different index URLs / package names). Users who run `pip install nufi` will silently get CPU-only execution with no warning, despite expecting GPU acceleration. Consider one of: (a) making torch an optional dependency (e.g., `[project.optional-dependencies]` with `gpu = ["torch>=2.0.0"]`) so users explicitly opt in; (b) adding a runtime check that warns if `torch.cuda.is_available()` returns `False`; or (c) clarifying in the description that a CUDA-capable torch must be installed separately.

  dependencies = [
      "numpy>=1.20.0",
      "scipy>=1.6.0",
      "pandas>=1.2.0",
      "scikit-learn>=1.0.0",
      "torch>=2.0.0"
  ]

\+ 

* [project.optional-dependencies]
* gpu = ["torch>=2.0.0"]  # Install with: pip install nufi[gpu]
]]></description>
  </task>
</tasklist>
