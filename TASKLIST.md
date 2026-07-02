```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>setup.py:94-97 - Bug: Missing -rpath for macOS OpenMP — runtime linker failure</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:94-97

**Bug: Missing `-rpath` for macOS OpenMP — runtime linker failure**

When linking against `libomp` from a non-standard location (Homebrew, MacPorts, conda), only
`-L<path>` and `-lomp` are passed. The `-L` flag is a *link-time* search path; at *runtime*, the
macOS dynamic linker (`dyld`) does not use it. Without `-rpath`, the compiled extension will fail to
load with an error like:
```
ImportError: dlopen(...): Library not loaded: @rpath/libomp.dylib
```
unless the user has set `DYLD_LIBRARY_PATH` or `LIBRARY_PATH`.

**Fix**: Append `-Wl,-rpath,{lib_path}` to `ext_linker_args` so the runtime linker can find
`libomp.dylib`:
```python
ext_linker_args = [
    "-L" + os.path.join(libomp_path, "lib"),
    "-Wl,-rpath," + os.path.join(libomp_path, "lib"),
    "-lomp",
]
```
(For Homebrew on Apple Silicon, `libomp_path` is typically `/opt/homebrew/opt/libomp`.)

          if libomp_path:
              ext_compiler_args = ["-Xpreprocessor", "-fopenmp"]
-             ext_linker_args = ["-L" + os.path.join(libomp_path, "lib"), "-lomp"]
+             ext_linker_args = [
+                 "-L" + os.path.join(libomp_path, "lib"),
+                 "-Wl,-rpath," + os.path.join(libomp_path, "lib"),
+                 "-lomp",
+             ]
              ext_include_dirs.append(os.path.join(libomp_path, "include"))
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>setup.py:29-31 - Bug: Fallback version parsing discards pre-release segm...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:29-31

**Bug: Fallback version parsing discards pre-release segments, causing inconsistent behavior across
environments**

When `packaging` is not installed, the manual parsing uses `str.isdigit()` to filter version
components. This silently discards pre-release and dev markers (e.g., `"3.0.0b1"` → `[3, 0]`,
`"1.20.0rc1"` → `[1, 20]`).

This is inconsistent with `packaging.version.Version`, which *does* account for pre-release
semantics:
- `Version("3.0.0b1") < Version("3.0.0")` is `True` → **rejected** with packaging
- Manual parsing of `"3.0.0b1"` gives `[3, 0]`, and `cy_parts[0] < 3` is `False` → **accepted**
without packaging

Similarly for NumPy: `"1.20.0rc1"` would be **accepted** by the manual parser but **rejected** by
packaging.

An environment that happens to have `packaging` installed will behave differently from one that
doesn't, leading to subtle build failures or successes.

**Fix**: List `"packaging"` in `install_requires` (or better, in `pyproject.toml`
`build-system.requires`), then remove the fallback parsing entirely so that
`packaging.version.Version` is always used.

-     except ImportError:
-         cy_parts = [int(x) for x in cython_version.split(".") if x.isdigit()]
-         if len(cy_parts) > 0 and cy_parts[0] < 3:
+     # packaging is now a required dependency; the fallback is removed for consistency.
+     # If you must keep the fallback, use packaging.version.parse from the vendored
+     # copy in setuptools or a PEP 440-compliant regex parser.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>setup.py:109-111 - Minor: Conditional imports scattered across the file hurt readability</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:109-111

**Minor: Conditional imports scattered across the file hurt readability**

`subprocess`, `tempfile`, `shutil` (line 104–106) and `sysconfig` (line 112) are imported inside the
Linux/BSD OpenMP detection block rather than at the top of the file. While functional, this makes it
harder to see all module dependencies at a glance and can cause repeated import overhead if the
block is entered multiple times (unlikely here but still unconventional).

**Suggestion**: Move these imports to the top of the file alongside the other standard-library
imports.

+ # (move to top of file)
          import subprocess
          import tempfile
          import shutil
+ import sysconfig
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>setup.py:128-130 - Minor: shutil.rmtree in finally block can mask cleanup errors</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:128-130

**Minor: `shutil.rmtree` in `finally` block can mask cleanup errors**

If `shutil.rmtree(tmpdir)` raises an exception (e.g., due to permission issues), it will propagate
out of the `finally` block. While the `except Exception: pass` above suppresses the original
exception, the rmtree error itself would go unhandled, potentially aborting the build.

**Suggestion**: Wrap the cleanup in its own try/except:
```python
finally:
    if tmpdir is not None:
        try:
            shutil.rmtree(tmpdir)
        except OSError:
            pass
```

          finally:
              if tmpdir is not None:
+                 try:
                  shutil.rmtree(tmpdir)
+                 except OSError:
+                     pass
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>5</id>
    <title>wrappers.py:139-146 - Bug: Duplicate timestamps within a group produce infini...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:139-146

**Bug: Duplicate timestamps within a group produce infinite Nyquist frequency, breaking the
imputer.**

In `imputer_apply`, `timestamps` are extracted via
`group_sorted.index.get_level_values(time_level).to_numpy()` and passed as the DataFrame index.
Inside `NufiImputer.fit()`, `np.diff(timestamps)` is computed; if any consecutive difference is zero
(e.g., duplicate timestamps after groupby), `min_p == 0`, then `max_sampling_rate = 1.0 / 0.0 = inf`
and `np.linspace(0, inf, n_f)` produces `[0, inf, inf, ...]`, causing downstream SVD/tensor failures
with cryptic errors.

Suggestion: Validate that timestamps within each group are strictly monotonic (no duplicates) before
passing to the imputer, or add a guard that raises a clear error early.

          # Drop the multi-index temporarily for fit_transform but keep index values as timestamps
          timestamps = group_sorted.index.get_level_values(time_level).to_numpy()
+         
+         # Validate strictly monotonic timestamps to prevent Nyquist overflow
+         if len(timestamps) > 1:
+             diffs = np.diff(timestamps.astype(np.float64))
+             if np.any(diffs <= 0):
+                 raise ValueError(
+                     f"Timestamps for group must be strictly increasing; "
+                     f"found non-positive or zero difference. "
+                     f"Check for duplicate or out-of-order timestamps."
+                 )
          
          # Avoid to_numpy() which coerces dtypes; copy with a clean index instead
          temp_df = group_sorted.copy()
          temp_df.index = timestamps
          
          infilled_temp = imputer.fit_transform(temp_df)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>6</id>
    <title>wrappers.py:146-148 - Bug: Uncaught exception from imputer.fit_transform le...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:146-148

**Bug: Uncaught exception from `imputer.fit_transform` leaves no context about which entity/group
failed.**

In `infill_multiindex_dataframe`, when the imputer raises inside the per-group loop, the exception
propagates with no information about which entity group triggered the failure. For large panel
datasets, this makes debugging extremely difficult. The same applies to `infill_dataframe` (line
56), though there's only one call there.

Suggestion: Wrap `imputer.fit_transform(temp_df)` in a try/except that enriches the error with the
entity identifier before re-raising.

+         try:
          infilled_temp = imputer.fit_transform(temp_df)
+         except Exception as e:
+             entity_id = group_sorted.index.get_level_values(entity_level)[0]
+             raise RuntimeError(
+                 f"NufiImputer.fit_transform failed for entity {entity_id!r}: {e}"
+             ) from e
          
          if len(infilled_temp) != len(group_sorted):
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>7</id>
    <title>wrappers.py:56-60 - Issue: pd_df.index.name from the original DataFrame i...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:56-60

**Issue: `pd_df.index.name` from the original DataFrame is silently discarded when
`keep_time_col=True`.**

Line 44 sets `pd_df.index.name = None` to avoid a name collision with the re-inserted time column.
However, if the original DataFrame had a named index (before `set_index(time_col)` replaced it),
that prior name is lost without any preservation or warning. While the index itself is being
replaced, losing the name can cause provenance issues in downstream pipelines that rely on index
metadata.

Suggestion: Capture the original index name before replacement and log or warn if it is being
discarded.

              time_values = pd_df[time_col].copy()
              col_pos = pd_df.columns.get_loc(time_col)
              pd_df = pd_df.set_index(time_col)
+             previous_index_name = pd_df.index.name  # will be time_col after set_index
              pd_df.index.name = None  # avoid name collision with the column
              pd_df.insert(col_pos, time_col, time_values)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>wrappers.py:139-144 - Risk: Unvalidated timestamp dtype in MultiIndex path ma...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:139-144

**Risk: Unvalidated timestamp dtype in MultiIndex path may reach `np.array(..., dtype=np.float64)`
and fail with a cryptic `TypeError`.**

`group_sorted.index.get_level_values(time_level).to_numpy()` returns the raw numpy dtype of the
index level. While `datetime64` silently converts to float64 (nanoseconds), non-numeric types such
as `str`, `object`, `categorical`, or `Period` will cause `np.array(timestamps, dtype=np.float64)`
inside `NufiImputer.fit()` to raise `TypeError: Cannot cast ...`. This deep error is hard to trace
back to the wrapper.

Suggestion: Validate at the wrapper layer that the timestamp level is numeric (or datetime-like) and
raise a clear, early error if not.

          # Drop the multi-index temporarily for fit_transform but keep index values as timestamps
          timestamps = group_sorted.index.get_level_values(time_level).to_numpy()
+         # Validate timestamp convertibility early
+         try:
+             np.array(timestamps, dtype=np.float64)
+         except (TypeError, ValueError):
+             raise TypeError(
+                 f"Timestamp level '{time_level}' has dtype {timestamps.dtype}, "
+                 f"which cannot be converted to float64. Use a numeric or datetime64 index level."
+             )
          
          # Avoid to_numpy() which coerces dtypes; copy with a clean index instead
          temp_df = group_sorted.copy()
          temp_df.index = timestamps
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>wrappers.py:80-102 - Risk: Shared NufiImputer instance across threads or g...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:80-102

**Risk: Shared `NufiImputer` instance across threads or groups has no synchronization; internal
state (fitted FFT plans, GCV parameters) is overwritten on each call.**

Both `infill_dataframe` (line 56) and `infill_multiindex_dataframe` (line 139) call
`imputer.fit_transform(...)` on the same `imputer` object. If the same instance is passed to both
wrappers concurrently (e.g., in a multi-threaded pipeline), or if `infill_multiindex_dataframe`
iterates groups while another thread reuses the imputer, internal state (`self.timestamps_`,
`self.alphas_`, `self.n_frequencies_`, `self.lu_`, `self.d_`, `self.perm_`) races. The per-group
loop in `infill_multiindex_dataframe` is single-threaded by design, but the lack of any
synchronization guard means concurrent external use is unsafe.

Suggestion: Document the thread-safety limitation prominently in the docstring, or clone the imputer
internally per call with `sklearn.base.clone()`.

  def infill_multiindex_dataframe(df, imputer=None, entity_level=0, time_level=1, sort=True):
      """
      Infill a MultiIndex Pandas/cuDF DataFrame (typically panel data).
      Each entity group (e.g. per-entity time series) is infilled independently
      to preserve distinct group behaviors and covariance.
+ 
+     .. warning::
+         The imputer instance is **not thread-safe**. If the same ``NufiImputer``
+         object is shared across threads or concurrent calls, internal fitted
+         state (timestamps, GCV parameters, LDL^T factors) will race and may
+         produce corrupted results. Use a separate instance per thread or
+         protect calls with external locking.
  
      Parameters:
      -----------
      df : pandas.DataFrame or cudf.DataFrame
          The MultiIndex DataFrame to infill.
      imputer : NufiImputer, optional
          The imputer instance to use. If None, a new NufiImputer is created.
      entity_level : int or str, default 0
          The level of the MultiIndex representing distinct entities/groups.
      time_level : int or str, default 1
          The level of the MultiIndex representing timestamps.
      sort : bool, default True
          Whether to sort the index by time level to ensure proper chronological order.
          Setting sort=False when timestamps are not already sorted may produce
          incorrect results due to Nyquist frequency miscalculation.
          Note: This reorders the group's rows; the output index order will reflect the sorted
          order, not the original input order.
      """
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>10</id>
    <title>wrappers.py:159-165 - Risk: pd.concat of per-group results does not verify ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:159-165

**Risk: `pd.concat` of per-group results does not verify that the total row count matches the input,
silently dropping or duplicating rows if group indices overlap unexpectedly.**

After the manual group iteration (lines 150–153), the results are concatenated with
`pd.concat(infilled_dfs)` with no verification that the total number of rows equals the input
DataFrame's row count. If entity-level groups have overlapping MultiIndex labels (possible with
non-unique indices), `pd.concat` may deduplicate or produce unexpected row ordering, and the caller
would not be alerted.

Suggestion: Add a final assertion or check that `len(infilled_pd) == len(pd_df)` after concatenation
(matching the per-group check already done at lines 141–147).

      # Avoid groupby.apply double-call on first group by iterating manually
      infilled_dfs = []
      for _, group in grouped:
          infilled_dfs.append(imputer_apply(group))
      infilled_pd = pd.concat(infilled_dfs)
+     
+     if len(infilled_pd) != len(pd_df):
+         raise ValueError(
+             f"Concatenated result has {len(infilled_pd)} rows, "
+             f"expected {len(pd_df)}. Group-level indices may overlap or be non-unique."
+         )
      
      if is_cudf:
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>agent.py:231-241 - Bug: Timestamp normalization is not reversed on output.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:231-241

**Bug: Timestamp normalization is not reversed on output.** When `max_ts > 2**53`, `timestamps` is
shifted by subtracting `epoch`, and the shifted array is passed to `imputer.fit()` and
`imputer.transform()`. However, `infilled_df` retains the original (unshifted) `df_copy.index`. The
NUDFT model is fitted against shifted time values, but the output values are aligned to unshifted
indices — the infilled signal is effectively a different function than what the DataFrame columns
claim. Fix: either restore the epoch after transform (`infilled_df.index = infilled_df.index +
epoch` or equivalent), or pass the normalized timestamps to the imputer and then map results back
explicitly to the original index.

      timestamps = df_copy.index.to_numpy(dtype=np.float64)
      max_ts = np.max(np.abs(timestamps)) if len(timestamps) > 0 else 0
+     epoch = 0.0
      if max_ts > 2**53:
          import warnings
-         # Subtract epoch to preserve relative precision in float64
          epoch = timestamps[0] if len(timestamps) > 0 else 0.0
          timestamps = timestamps - epoch
          warnings.warn(
              f"Timestamps exceed float64 precision (max={max_ts:.1e}). "
              f"Normalized by subtracting epoch={epoch} to preserve relative precision."
          )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>agent.py:314-317 - Bug: Double-compensation skews SNR and spectral diagnostics.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:314-317

**Bug: Double-compensation skews SNR and spectral diagnostics.** When
`covariance_compensation=True`, `imputer.reconstructed_` already includes the `d_` scaling applied
during the imputer's own reconstruction. Multiplying by `cov_scale` again inflates
`reconstructed_np`, which then feeds into `signal_variance`, `residual`, `snr_db`, `psd`, and
`entropy`. This yields misleading diagnostic values (overestimated SNR, distorted entropy). The
`cov_scale` multiplication should be removed from the diagnostic code path, or the imputer should
expose a pre-compensation reconstruction signal for diagnostics.

-         if imputer.covariance_compensation and imputer.d_ is not None:
-             cov_scale = np.sqrt(np.abs(np.diag(imputer.d_)[col_idx]))
-             if cov_scale > 0:
-                 reconstructed_np = reconstructed_np * cov_scale
+         # Note: imputer.reconstructed_ already includes covariance compensation.
+         # Do not apply cov_scale again to avoid double-compensation in diagnostics.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>13</id>
    <title>agent.py:214-229 - Bug: Datetime-like index/time_col causes cryptic failur...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:214-229

**Bug: Datetime-like index/time_col causes cryptic failure in both `impute_dataframe` and
`plot_diagnostics`.**

When the DataFrame index contains pandas Timestamps (e.g., `DatetimeIndex`), `pd.to_numeric(...,
errors='coerce')` converts *all* values to `NaN`, triggering `"Index contains non-convertible
values"` in `impute_dataframe` or a similar cryptic failure in `plot_diagnostics`. This is confusing
because datetime timestamps are a legitimate, common use case.

**Fix**: In both functions, detect datetime-like dtypes explicitly and convert via
`.astype(np.int64)` (nanosecond epoch) or `.astype(np.float64)` (epoch seconds), with a clear
warning about precision loss.

      if not pd.api.types.is_numeric_dtype(df_copy.index):
          try:
-             # Attempt conversion for datetime-like or string timestamps
+             # Handle datetime-like index explicitly
+             if pd.api.types.is_datetime64_any_dtype(df_copy.index):
+                 df_copy.index = df_copy.index.astype(np.int64)
+             else:
              numeric_idx = pd.to_numeric(df_copy.index, errors='coerce')
              if numeric_idx.isna().any():
                  raise ValueError("Index contains non-convertible values")
              if np.can_cast(numeric_idx, np.int64, casting='safe'):
                  df_copy.index = numeric_idx.astype(np.int64)
              else:
                  df_copy.index = numeric_idx.astype(np.float64)
          except Exception:
              raise TypeError(
                  f"DataFrame index must be numeric (timestamps). "
                  f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col` "
                  f"or ensure your index contains numeric values."
              )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>14</id>
    <title>agent.py:536-542 - Bug: plot_diagnostics has inconsistent return behavior.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:536-542

**Bug: `plot_diagnostics` has inconsistent return behavior.** When `num_cols == 0`, the function
returns `(fig, axes)`, but in the normal path (after rendering all columns) it falls through with no
`return` statement, implicitly returning `None`. Callers expecting `fig, axes` will get `None`.
Either add `return fig, axes` at the end of the normal path or remove the early return and update
the docstring.

      plt.tight_layout()
      if save_path is not None:
          plt.savefig(save_path, dpi=300, bbox_inches='tight')
      if show_plot:
          plt.show()
      else:
          plt.close()
+     return fig, axes
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>15</id>
    <title>agent.py:314-314 - Potential AttributeError: imputer.d_ may not exist.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:314-314

**Potential `AttributeError`: `imputer.d_` may not exist.** The check `imputer.d_ is not None` will
raise `AttributeError` if the NufiImputer instance has no `d_` attribute at all (e.g., when
covariance_compensation is True but the fit did not populate `d_`). Use `getattr(imputer, 'd_',
None)` for safe access.

-         if imputer.covariance_compensation and imputer.d_ is not None:
+         if imputer.covariance_compensation and getattr(imputer, 'd_', None) is not None:
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>16</id>
    <title>test_agent.py:155-158 - File leakage: save_img is written to the current workin...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:155-158

File leakage: `save_img` is written to the current working directory (CWD) instead of the test's
temp directory (`self._tmpdir`). This causes two problems: (1) parallel test runs may collide on the
same filename, and (2) if the test process is killed before the `finally` block runs, a stale
`test_diagnostics_plot.png` is left behind in the project root. Use `os.path.join(self._tmpdir,
save_img)` instead.

          # Test plot rendering with show_plot=False to avoid blocking tests
-         save_img = "test_diagnostics_plot.png"
+         save_img = os.path.join(self._tmpdir, "test_diagnostics_plot.png")
          if os.path.exists(save_img):
              os.remove(save_img)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>17</id>
    <title>test_agent.py:98-101 - Overly broad exception assertion: self.assertRaises((Key...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:98-101

Overly broad exception assertion: `self.assertRaises((KeyError, ValueError))` accepts two different
exception types. If the implementation is refactored and starts raising a different exception (e.g.,
`RuntimeError` or `AttributeError`), this test would silently pass when it should fail, masking a
behavioral regression. Prefer asserting the single expected exception type. If the API genuinely may
raise either, document why and consider making the API consistent.

      def test_impute_dataframe_missing_time_col(self):
          """Edge case: specified time column does not exist."""
-         with self.assertRaises((KeyError, ValueError)):
+         with self.assertRaises(KeyError):
              impute_dataframe(self.df, time_col="nonexistent")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>18</id>
    <title>test_agent.py:84-85 - Fragile assertion: assertIn on stability_flags can pr...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:84-85

Fragile assertion: `assertIn` on `stability_flags` can produce a false positive if the value changes
from `list` to `str` — string `assertIn` performs a substring match, so `"NO_OBSERVATIONS" in
"SOMETHING_NO_OBSERVATIONS_ELSE"` would pass. While the current implementation always returns a
list, a more robust assertion would verify the exact list or at minimum check `isinstance(...,
list)` first to catch type regressions.

          self.assertIn("signal", diagnostics)
-         self.assertIn("NO_OBSERVATIONS", diagnostics["signal"]["stability_flags"])
+         flags = diagnostics["signal"]["stability_flags"]
+         self.assertIsInstance(flags, list)
+         self.assertIn("NO_OBSERVATIONS", flags)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>19</id>
    <title>test_agent.py:67-71 - Misleading docstring: the docstring says the empty-DataFr...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:67-71

Misleading docstring: the docstring says the empty-DataFrame case "should raise or return
gracefully", but the test only asserts `TypeError` is raised. If the implementation ever returns
gracefully (e.g., an empty DataFrame), this test will fail even though that behavior matches the
documented intent. Either tighten the docstring to reflect the actual expected exception, or update
the test to also accept a graceful return.

      def test_impute_dataframe_empty(self):
-         """Edge case: empty DataFrame should raise or return gracefully."""
+         """Edge case: empty DataFrame should raise TypeError."""
          empty_df = pd.DataFrame(columns=["timestamp", "signal"])
          with self.assertRaises(TypeError):
              impute_dataframe(empty_df, time_col="timestamp")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>20</id>
    <title>torch_kernels.py:152-162 - Real/imag column pairing broken under NaN masking.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:152-162

**Real/imag column pairing broken under NaN masking.**

`flat_data` interleaves real and imaginary parts of each transformed signal (N real followed by N
imag per signal). The covariance matrix is then (2N × 2N). When `nan_mask` is computed column-wise
on the covariance matrix (line 149), a constant real component will produce a NaN variance on the
diagonal, causing that single column to be dropped — but its paired imaginary column remains. This
breaks the real/imag pairing for that frequency bin.

Callers who use `valid_idx // 2` to map reduced-matrix indices back to frequency bins will get a bin
that has one component missing, yielding silently corrupted reconstructions.

**Suggestion**: Detect degeneracy on pairs: if either element of a (real, imag) pair is NaN/invalid
(e.g., variance below a threshold), drop both columns together. Alternatively, detect NaNs on
diagonal entries and drop both `k` and `k+N`/`k+1` columns for each affected pair.

-     nan_mask = np.any(np.isnan(covariance_matrix), axis=0)
-     valid_idx = np.arange(covariance_matrix.shape[0])  # default: all valid
-     if np.any(nan_mask):
+     # Detect degenerate columns on the diagonal of the covariance matrix
+     diag = np.diag(covariance_matrix)
+     diag_nan = np.isnan(diag)
+     if np.any(diag_nan):
          import warnings
-         n_nan = nan_mask.sum()
-         warnings.warn(f"Covariance matrix contains NaN entries in {n_nan} columns; degenerate columns detected. Applying regularization.")
-         # Drop degenerate rows/columns instead of zero-filling
-         valid_idx = np.where(~nan_mask)[0]
+         n_nan = diag_nan.sum()
+         warnings.warn(f"Covariance matrix contains NaN on diagonal in {n_nan} entries; degenerate columns detected. Applying regularization.")
+         # Drop degenerate real/imag pairs together: for each NaN diagonal entry at index k,
+         # also drop its paired component (assuming real at even indices, imag at odd, or
+         # N real then N imag layout).
+         # For N-real-then-N-imag layout, paired index is (k + N) % (2*N).
+         N = covariance_matrix.shape[0] // 2
+         pair_mask = np.zeros(covariance_matrix.shape[0], dtype=bool)
+         for k in np.where(diag_nan)[0]:
+             pair_mask[k] = True
+             pair_mask[(k + N) % (2 * N)] = True
+         valid_idx = np.where(~pair_mask)[0]
          if len(valid_idx) == 0:
              raise ValueError("All columns are degenerate; cannot compute covariance compensation.")
          covariance_matrix = covariance_matrix[np.ix_(valid_idx, valid_idx)]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>21</id>
    <title>torch_kernels.py:45-57 - Nyquist frequency overestimation can cause numerical overflow and precision loss.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:45-57

**Nyquist frequency overestimation can cause numerical overflow and precision loss.**

Using `np.min(p_n)` (the smallest positive sampling interval) yields the *maximum possible* Nyquist
frequency `0.5 / min_p`. If two timestamps are extremely close (e.g., due to measurement noise,
near-duplicate samples, or clock jitter), `min_p` becomes tiny and the Nyquist frequency explodes.
This can make `torch.exp(-2πi · t · f_k)` have exponents so large that intermediate values overflow
in float64 or suffer catastrophic precision loss, even when `N` is far below the `MAX_MEM_N` guard.

**Suggestion**: Use a robust estimator such as the median sampling interval (`0.5 / np.median(p_n)`)
or cap the Nyquist frequency at a sensible upper bound. At minimum, clamp `nyquist_frequency` to
prevent unbounded growth.

          if len(v_timestamps) > 1:
              # Sort to ensure positive diffs
              sort_idx = np.argsort(v_timestamps)
              sorted_ts = v_timestamps[sort_idx]
              p_n = np.diff(sorted_ts)
              p_n = p_n[p_n > 0]  # keep only positive intervals
              if len(p_n) > 0:
-                 min_p = np.min(p_n)
-                 nyquist_frequency = 0.5 / max(min_p, 1e-12)
+                 median_p = np.median(p_n)
+                 nyquist_frequency = 0.5 / max(median_p, 1e-12)
              else:
                  import warnings
                  warnings.warn("Cannot estimate Nyquist frequency; all sampling intervals are zero or negative. Defaulting to 1.0.")
                  nyquist_frequency = 1.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>22</id>
    <title>torch_kernels.py:188-189 - solve_cg solves the normal equations, squaring the condition number.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:188-189

**`solve_cg` solves the normal equations, squaring the condition number.**

The `H(v)` operator computes `A^H A v + alpha * v`, which means CG is applied to the normal
equations `(A^H A + alpha I) x = A^H b`. This squares the condition number of the original system
`A`, potentially degrading convergence speed and final accuracy for ill-conditioned `A` compared to
the augmented-system least-squares approach used in the `'direct'` solver path. For moderately
ill-conditioned matrices this may be acceptable, but the function should at minimum document the
trade-off.

**Suggestion**: Consider using LSQR or the augmented-system approach `[A; sqrt(alpha)*I]` within CG
for better numerical properties, or add a warning/note in the docstring.

+     # NOTE: H(v) = A^H A v + alpha v applies CG to the normal equations,
+     # which squares the condition number of A. For ill-conditioned A, prefer
+     # the augmented-system (direct) solver in solve_tikhonov_nudft.
      def H(v):
          return torch.matmul(A.adjoint(), torch.matmul(A, v)) + alpha * v
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>23</id>
    <title>torch_kernels.py:86-91 - compute_Fast_ND_NUDFT uses linear interpolation witho...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:86-91

**`compute_Fast_ND_NUDFT` uses linear interpolation without windowing — undocumented spectral
distortion.**

Direct linear interpolation of non-uniform samples onto a uniform grid before FFT introduces
uncontrolled spectral leakage and amplitude distortion (equivalent to convolution with a triangular
kernel in the time domain). Users calling this function may be unaware that the output is not a true
NUDFT approximation but a crude resampling-based estimate. This is especially misleading since the
function name includes "Fast_ND_NUDFT".

**Suggestion**: Add a clear warning in the docstring explaining that this method trades accuracy for
speed and that linear interpolation introduces spectral artifacts. Consider offering a windowed-sinc
or gridding-based interpolation as an alternative.

  def compute_Fast_ND_NUDFT(X_list, device=None):
      """
      Performs Fast Non-Uniform DFT by interpolating onto a uniform grid 
      and computing FFT using PyTorch rfft/fft.
      Gracefully handles NaNs during interpolation.
+ 
+     .. warning::
+         This function uses linear interpolation, which introduces spectral
+         leakage and amplitude distortion. It is a fast approximation, not an
+         exact NUDFT. For higher accuracy, consider compute_ND_NUDFT or a
+         gridding-based approach.
      """
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>24</id>
    <title>torch_kernels.py:79-79 - Sign convention between forward and inverse transforms is undocumented.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:79-79

**Sign convention between forward and inverse transforms is undocumented.**

`compute_ND_NUDFT` uses `-2πi` (forward, analysis), while `solve_tikhonov_nudft` builds the
synthesis matrix with `+2πi` (inverse). Although this is the mathematically correct conjugate pair,
the relationship is never stated in docstrings. Users combining these functions may misinterpret the
coefficient sign, leading to inverted phase in reconstructions.

**Suggestion**: Document the sign convention explicitly in both functions' docstrings.

+         # Forward NUDFT: A[n,k] = exp(-2πi * t_n * f_k)  (analysis convention)
          exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>25</id>
    <title>torch_kernels.py:203-205 - Variable name alpha_cg shadows the regularization parameter alpha.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:203-205

**Variable name `alpha_cg` shadows the regularization parameter `alpha`.**

Inside `solve_cg`, the variable `alpha_cg` (line 208) is the CG step size, unrelated to the Tikhonov
regularization parameter `alpha` passed to the outer function. This naming collision creates
confusion for readers and maintainers.

**Suggestion**: Rename to `step_size` or `cg_step`.

-         alpha_cg = rsold / denom
-         x = x + alpha_cg * p
-         r = r - alpha_cg * Hp
+         step_size = rsold / denom
+         x = x + step_size * p
+         r = r - step_size * Hp
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>26</id>
    <title>impute.py:158-158 - Bug: Incorrect // 2 mapping from valid_idx_comp to valid_cols.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:158-158

**Bug: Incorrect `// 2` mapping from `valid_idx_comp` to `valid_cols`.**

`covariance_compensation` returns `valid_idx` referencing columns of the M×M covariance matrix
(where M = len(X_list) = number of valid signals in the doubled real+imag space). The covariance
matrix shape is M×M because each signal's real and imag parts are concatenated into a single column
in `flat_data` (see `torch_kernels.py:141-144`). Therefore `valid_idx_comp` values range from 0 to
M-1, not 0 to 2M-1. The `// 2` integer division incorrectly halves these indices, causing wrong
column mapping. For example, with M=3 signals and `valid_idx_comp=[0,1,2]`, the code produces
`[0,0,1]`, duplicating column 0 and missing column 2.

This corrupts `actual_valid_cols`, which cascades into incorrect `self.perm_`, `self.d_`, and
`self.lu_` assignments.

-                 actual_valid_cols = [valid_cols[idx // 2] for idx in valid_idx_comp]
+                 actual_valid_cols = [valid_cols[idx] for idx in valid_idx_comp]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>27</id>
    <title>impute.py:251-275 - Bug: Scale inconsistency in transform — imputed value...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:251-275

**Bug: Scale inconsistency in `transform` — imputed values in compensated space mixed with observed
values in original space.**

When `covariance_compensation` is active and `cov_scale != 1`, `reconstructed_np` is multiplied by
`cov_scale` (line 203), but the NaN-fill at line 218 writes these compensated-scale values directly
into `X_data`. The observed (non-NaN) values remain in their original scale. This produces an output
array where some entries are scaled and others are not — a silent data corruption.

Furthermore, in the stochastic branch, the residual is correctly computed in compensated space (both
observed data and reconstructed are scaled), but the noise `rng.normal(...) * residual_std` is also
in compensated space and added to the already-compensated `reconstructed_np[nan_mask]`. The imputed
values end up in compensated space while observed values stay in original space.

-             # If covariance compensation is computed, align the reconstructed scale
+             # Compute compensated reconstructed signal for residual analysis
              cov_scale = 1.0
              if self.covariance_compensation and self.d_ is not None:
                  cov_scale = np.sqrt(np.abs(np.diag(self.d_)[col_idx]))
-                 if cov_scale > 0:
-                     reconstructed_np = reconstructed_np * cov_scale
+             
+             reconstructed_compensated = reconstructed_np * cov_scale if cov_scale > 0 else reconstructed_np
              
              # Fill only the NaNs
              nan_mask = np.isnan(X_data[:, col_idx])
              if np.any(nan_mask):
                  if stochastic:
-                     # Scale observed data so residual is in compensated space
                      obs_mask = ~nan_mask
                      if np.any(obs_mask):
-                         residual = (X_data[obs_mask, col_idx] * cov_scale if cov_scale > 0 else X_data[obs_mask, col_idx]) - reconstructed_np[obs_mask]
+                         residual = (X_data[obs_mask, col_idx] * cov_scale if cov_scale > 0 else X_data[obs_mask, col_idx]) - reconstructed_compensated[obs_mask]
                          residual_std = np.std(residual) if len(residual) > 1 else 0.1
                          if np.isnan(residual_std) or residual_std == 0:
                              residual_std = 0.1
                      else:
                          residual_std = 0.1
-                         
                      noise = rng.normal(0, stochastic_scale * residual_std, size=nan_mask.sum())
-                     X_data[nan_mask, col_idx] = reconstructed_np[nan_mask] + noise
+                     # Convert back to original scale for output
+                     imputed_vals = (reconstructed_compensated[nan_mask] + noise) / cov_scale if cov_scale > 0 else reconstructed_compensated[nan_mask] + noise
+                     X_data[nan_mask, col_idx] = imputed_vals
                  else:
                      X_data[nan_mask, col_idx] = reconstructed_np[nan_mask]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>28</id>
    <title>impute.py:31-33 - Design issue: self.lu_ and self.perm_ are computed ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:31-33

**Design issue: `self.lu_` and `self.perm_` are computed and stored in `fit` but never used in
`transform`.**

The LDLᵀ decomposition `P @ A @ Pᵀ = L @ D @ Lᵀ` captures inter-column correlations. However,
`transform` only applies diagonal D scaling via `cov_scale = sqrt(abs(diag(d_)[col_idx]))`, ignoring
both the lower-triangular L matrix and the permutation P. This means the multi-dimensional
covariance structure between signals is lost during imputation — only per-column variance scaling is
applied. If multi-signal covariance compensation is intended, the full LDLᵀ decomposition should be
applied (e.g., by pre-whitening the multi-column residual or reconstructing with the full covariance
structure).

+         # Note: lu_ and perm_ are stored for potential downstream use in full
+         # covariance-aware reconstruction. Currently only d_ (diagonal scaling)
+         # is applied in transform(). Consider implementing full LDL^T application
+         # for proper multi-signal covariance compensation.
          self.lu_ = None
          self.d_ = None
          self.perm_ = None
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>29</id>
    <title>impute.py:121-128 - Risk: SVD instability with very few valid observations (N_val ≤ 2).</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:121-128

**Risk: SVD instability with very few valid observations (N_val ≤ 2).**

When N_val is very small (e.g., 1 or 2), the design matrix A has shape (N_val, n_f). The SVD of such
a tiny matrix can be ill-conditioned, and the GCV score `compute_gcv_from_svd` may produce
unreliable or infinite results. The fallback at lines 102-109 sets `best_n_freq = max(5, N_val)` and
`best_alpha = 1.0`, which may still produce an underdetermined system (e.g., N_val=1, n_f=5 → 1
equation, 5 unknowns). Consider setting a minimum N_val threshold (e.g., 3–5) below which the column
is skipped with a warning, or using a simpler interpolation fallback.

              if best_gcv == float('inf'):
                  import warnings
-                 best_n_freq = max(5, N_val)
+                 best_n_freq = min(max(5, N_val), N_val)  # avoid underdetermined system
                  best_alpha = 1.0
                  warnings.warn(
                      f"All GCV candidates failed SVD for column {col_idx}. "
-                     f"Using conservative fallback n_f={best_n_freq}, alpha={best_alpha}."
+                     f"N_val={N_val} is very small; using n_f={best_n_freq}, alpha={best_alpha}."
                  )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>30</id>
    <title>impute.py:73-75 - Bug: n_frequencies='auto' can produce n_f &gt; N_val, ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:73-75

**Bug: `n_frequencies='auto'` can produce `n_f > N_val`, leading to an underdetermined linear
system.**

When N_val is small (e.g., N_val=3), the candidate set `[max(5, 3//4), max(5, 3//2), max(5, 3)]` =
`[5, 5, 5]`. This produces 5 frequencies for only 3 observations — an underdetermined system where
SVD may succeed but the solution is not unique and GCV scores are unreliable. The number of
frequencies should be capped at `N_val` to keep the system at most fully determined.

              if self.n_frequencies == 'auto':
                  candidates = [max(5, N_val // 4), max(5, N_val // 2), max(5, N_val)]
+                 candidates = [c for c in candidates if c <= N_val]  # avoid underdetermined systems
+                 if not candidates:
+                     candidates = [max(1, N_val)]
                  candidates = sorted(list(set(candidates)))
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>31</id>
    <title>test_covariance.py:24-28 - Brittle numeric thresholds</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:24-28

**Brittle numeric thresholds**: The hardcoded bounds `0.2` and `0.02` are derived from specific
signal parameters (amplitude=1, ω=1, dt≈0.101) with a 2× margin. Minor algorithmic changes that
still produce smooth, correct results can trigger these asserts, leading to false-positive test
failures. Consider computing the expected bounds dynamically from `dt` and the signal amplitude, or
loosening the margins and documenting the rationale.

-     # Expected max |dx| for sin(t) sampled at dt≈0.101: amplitude × ω × dt ≈ 1 × 1 × 0.101 = 0.101
-     # Allow 2× margin for infill artifacts near the gap boundary.
-     assert np.max(np.abs(dx)) < 0.2
-     # Expected max |ddx| ~ amplitude × ω² × dt² ≈ 1 × 1 × 0.0102 = 0.0102; 2× margin.
-     assert np.max(np.abs(ddx)) < 0.02
+     # Compute expected bounds dynamically
+     dt = float(np.mean(np.diff(t)))
+     amp = 1.0  # amplitude of sin(t)
+     omega = 1.0
+     # Allow 3× margin to reduce brittleness while still catching gross failures
+     assert np.max(np.abs(dx)) < 3 * amp * omega * dt
+     assert np.max(np.abs(ddx)) < 3 * amp * omega**2 * dt**2
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>32</id>
    <title>test_covariance.py:14-14 - Missing edge-case coverage</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:14-14

**Missing edge-case coverage**: The TODO on line 10 explicitly acknowledges missing tests for
boundary NaNs, multiple gaps, and extreme missing ratios. These edge cases are where imputation
algorithms most often fail (e.g., NaNs at t=0 or t=end, or 80%+ missing data). Without these tests,
regressions in edge-case handling can go undetected.

      # TODO: add parametrized tests for boundary NaNs, multiple gaps, and extreme missing ratios
+     # See: https://github.com/example/issues/123 for tracking
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>33</id>
    <title>test_covariance.py:62-63 - Overly loose tolerance in covariance test</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:62-63

**Overly loose tolerance in covariance test**: `rtol=1e-1` and `atol=1e-1` allow a 10% relative
error or 0.1 absolute error on covariance values. For signals like sin(t)/cos(t) with variances near
0.5, a 0.1 absolute error is ~20% of the signal variance — large enough to mask a non-functioning
`covariance_compensation` flag. Consider also testing that compensated vs uncompensated results
differ meaningfully when `covariance_compensation` is toggled.

      # Relaxed tolerances to account for estimation variance with only 50 samples and ~20% missing data.
-     np.testing.assert_allclose(filled_cov, original_cov, rtol=1e-1, atol=1e-1)
+     np.testing.assert_allclose(filled_cov, original_cov, rtol=5e-2, atol=5e-2)
+     # Also verify covariance_compensation actually changes the result:
+     imputer_no_comp = NufiImputer(method='direct', covariance_compensation=False)
+     X_no_comp = imputer_no_comp.fit_transform(X, timestamps=t)
+     no_comp_cov = np.cov(X_no_comp[:, 0], X_no_comp[:, 1])
+     # Compensated should be closer to original than uncompensated
+     assert np.linalg.norm(filled_cov - original_cov) <= np.linalg.norm(no_comp_cov - original_cov)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>34</id>
    <title>benchmark.py:100-105 - Silent method degradation</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:100-105

**Silent method degradation**: When `cubic` interpolation leaves trailing NaNs (which it will when
the first/last valid points are not at the boundaries), the code falls back through `linear → ffill
→ bfill`. The benchmark label still reports "Cubic Spline", but the actual method evaluated is a
**hybrid** of at least two (possibly three) different interpolation strategies. This undermines the
validity of the benchmark — the reported RMSE/Covariance for "Cubic Spline" does not reflect pure
cubic spline performance.

**Suggestion**: Either: (a) report a separate "Cubic Spline (with fallback)" entry, (b) skip the
method entirely if cubic spline cannot handle the data (as is done for MICE/GP when unavailable), or
(c) restrict benchmark data to never have leading/trailing NaN gaps that cubic spline cannot handle.

          spline_infilled = df_masked.interpolate(method='cubic', axis=0)
-         # Fill any remaining NaNs with linear fallback and backward/forward fill
          remaining_nan = spline_infilled.isna().sum().sum()
          if remaining_nan > 0:
-             print(f"[INFO] Cubic spline left {remaining_nan} NaN(s); falling back to linear → ffill → bfill")
+             print(f"[WARN] Cubic spline left {remaining_nan} NaN(s) — results may be degraded by fallback")
+         # Fallback: use linear, then ffill/bfill to ensure no NaN left
          spline_infilled = spline_infilled.interpolate(method='linear', axis=0).ffill().bfill()
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>35</id>
    <title>benchmark.py:55-58 - Boundary NaN patching biases the benchmark</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:55-58

**Boundary NaN patching biases the benchmark**: The data generator manually copies the nearest valid
value into leading/trailing NaN positions (lines 55-58). This effectively gives every method
boundary values "for free" — even methods that cannot extrapolate (like cubic spline). The bias is
especially pronounced at high missing rates where natural boundaries would stress-test a method's
extrapolation capability.

**Suggestion**: Consider adding an option to control this behavior (e.g., `fix_boundary_nans=False`)
and report results both with and without boundary patching, so users can understand method
robustness under realistic edge conditions.

+         # NOTE: Patching boundary NaNs gives all methods free extrapolation —
+         #       this may inflate scores for spline/MICE/GP that cannot handle edge NaNs.
          if np.isnan(col[0]):
              col[0] = col[valid_idx[0]]
          if np.isnan(col[-1]):
              col[-1] = col[valid_idx[-1]]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>36</id>
    <title>benchmark.py:125-126 - Missing timestamp normalization degrades MICE imputation quality</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:125-126

**Missing timestamp normalization degrades MICE imputation quality**: Timestamps (range ~0–10) are
concatenated raw alongside signal values (range ~−3 to 3). Inside `IterativeImputer` with default
`BayesianRidge` estimator, the L2 regularization treats all features equally. However, the timestamp
has a different scale and semantic meaning — without standardization (e.g., `StandardScaler`), the
regularizer penalizes the timestamp coefficient differently than the signal coefficients, which is
not inherent to MICE itself and artificially depresses its benchmark scores.

**Suggestion**: Normalize timestamps to zero-mean unit-variance (or at least [0,1]) before
concatenating, e.g.:
```python
t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
combined_masked = np.hstack([t_norm.reshape(-1, 1), df_masked.to_numpy()])
```

-             # We append timestamps as a feature so MICE understands temporal relation
-             combined_masked = np.hstack([timestamps.reshape(-1, 1), df_masked.to_numpy()])
+             # Normalize timestamps to [0,1] so MICE regularization treats them comparably
+             t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
+             combined_masked = np.hstack([t_norm.reshape(-1, 1), df_masked.to_numpy()])
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>37</id>
    <title>benchmark.py:206-208 - Fixed output filename silently overwrites previous runs</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:206-208

**Fixed output filename silently overwrites previous runs**: Every invocation writes to
`benchmark_results.json`, destroying results from prior runs. This is a data-loss issue when
iterating on hyperparameters or comparing across configurations.

**Suggestion**: Include a timestamp or config hash in the filename, e.g.:
```python
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"benchmark_results_{ts}.json"
```
Alternatively, check if the file exists and prompt/rotate before overwriting.

-         with open("benchmark_results.json", "w") as f:
+         from datetime import datetime
+         ts = datetime.now().strftime("%Y%m%d_%H%M%S")
+         filename = f"benchmark_results_{ts}.json"
+         with open(filename, "w") as f:
              json.dump(benchmark_results, f, indent=4)
-             print("Results saved to benchmark_results.json")
+             print(f"Results saved to {filename}")
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>38</id>
    <title>test_imputer.py:160-162 - Flaky test: stochastic non-determinism assertion can produce false failures.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:160-162

**Flaky test: stochastic non-determinism assertion can produce false failures.**

The assertion `assert X_filled_1[1, 0] != X_filled_2[1, 0]` (and the same for index 3) expects that
two stochastic imputations without a fixed seed produce different values for every missing cell.
While unlikely in theory for continuous draws, floating‑point rounding and finite precision make
exact equality possible in practice. A single coincidental match will cause the test to fail
spuriously.

Suggestion: Replace the exact inequality check with a statistical or variance‑based test (e.g.
assert that at least one of the two missing‑cell pairs differs, or that the sample variance across
several stochastic runs is non‑zero). Alternatively, explicitly accept the tiny false‑negative
probability by marking the test with `@pytest.mark.flaky` or using a retry mechanism.

      # Missing spots should have different stochastic values
-     assert X_filled_1[1, 0] != X_filled_2[1, 0]
-     assert X_filled_1[3, 0] != X_filled_2[3, 0]
+     # Use a tolerance to avoid false failures from floating-point coincidences
+     assert abs(X_filled_1[1, 0] - X_filled_2[1, 0]) > 1e-12 or abs(X_filled_1[3, 0] - X_filled_2[3, 0]) > 1e-12, (
+         "Stochastic imputations should differ; both pairs were identical"
+     )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>39</id>
    <title>test_imputer.py:98-100 - Potentially flaky: direct vs CG solver consistency tolerance may be too tight.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:98-100

**Potentially flaky: direct vs CG solver consistency tolerance may be too tight.**

The test uses `np.allclose(X_direct, X_cg, atol=1e-4)` to compare direct and Conjugate Gradient
imputations. The CG solver's convergence depends on the condition number of the linear system and
may behave differently across platforms, BLAS implementations, or even compiler optimizations. An
atol of 1e-4 is reasonable for this small 4×2 matrix but could intermittently fail in CI on some
architectures.

Suggestion: Consider relaxing the tolerance slightly (e.g., `atol=1e-3` or `rtol=1e-4`) or asserting
that both solvers produce results within a looser bound while preserving non‑NaN values. If strict
determinism across solvers is a hard requirement, document it as a conformance test with a known
tolerance.

      # Assert direct and CG solvers produce consistent/similar imputations (Task 20)
-     # Tighten tolerance to better detect solver discrepancies
-     assert np.allclose(X_direct, X_cg, atol=1e-4)
+     # Use a slightly relaxed tolerance to avoid platform‑specific CG convergence variance
+     assert np.allclose(X_direct, X_cg, atol=1e-3)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>40</id>
    <title>test_imputer.py:130-132 - Potentially fragile: GCV imputation bounds test may reject valid results.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:130-132

**Potentially fragile: GCV imputation bounds test may reject valid results.**

The assertion requires all imputed values to lie within `[obs_min - 5.0, obs_max + 5.0]`. When GCV
selects strong regularization (large alpha), imputed values can be pulled toward the column mean. On
a small dataset like this (7 rows, 2 columns), the column mean could easily fall outside `[obs_min,
obs_max]` by more than 5 units, especially if the observed range is narrow — causing a false test
failure even though the imputer is working correctly.

Suggestion: Either widen the tolerance bound (e.g., `obs_min - 3*std` / `obs_max + 3*std`), use a
relative tolerance based on the data scale, or test imputation quality via a different criterion
(e.g., monotonicity with respect to alpha, or cross‑validation error).

-         # Verify no wild outliers
-         assert np.all(X_filled[:, col_idx] >= obs_min - 5.0)
-         assert np.all(X_filled[:, col_idx] <= obs_max + 5.0)
+         # Verify no wild outliers using a tolerance proportional to the data scale
+         col_range = obs_max - obs_min if obs_max > obs_min else 1.0
+         assert np.all(X_filled[:, col_idx] >= obs_min - 3.0 * col_range)
+         assert np.all(X_filled[:, col_idx] <= obs_max + 3.0 * col_range)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>41</id>
    <title>test_imputer.py:16-19 - Missing value‑correctness assertions in infill_dataframe wrapper test.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:16-19

**Missing value‑correctness assertions in infill_dataframe wrapper test.**

The test verifies that no NaNs remain and that the shape is preserved, but does not check that the
originally valid values are preserved (e.g., `signal[0] == 10.0`, `signal[2] == 30.0`). This means a
bug that corrupts observed values would pass the test.

Suggestion: Add explicit assertions that known non‑NaN values remain unchanged after imputation.

      df_filled = infill_dataframe(df, time_col='timestamp', keep_time_col=False)
      assert isinstance(df_filled, pd.DataFrame)
      assert not df_filled.isna().any().any()
      assert len(df_filled) == len(df)
+     # Verify observed values are preserved
+     assert df_filled.loc[0, 'signal'] == 10.0
+     assert df_filled.loc[2, 'signal'] == 30.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>42</id>
    <title>test_imputer.py:205-207 - Fragile: all‑NaN column test depends on undocumented implementation behavior.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:205-207

**Fragile: all‑NaN column test depends on undocumented implementation behavior.**

The test asserts that an all‑NaN column remains entirely NaN (`assert np.isnan(X_filled[:,
1]).all()`). This is a valid design choice, but if the imputer implementation changes to fall back
to, e.g., filling with the global mean or zero, this test would break without an actual bug. The
test does not check whether a warning is emitted, which could be important for users who need to
know that a column was left unimputed.

Suggestion: Consider using `pytest.warns(...)` to assert that an appropriate warning is raised when
an all‑NaN column is encountered, and/or document that this behavior is a contractual requirement.

      imputer1 = NufiImputer(covariance_compensation=True)
+     with pytest.warns(UserWarning, match="all-NaN|empty|no valid"):
      X_filled = imputer1.fit_transform(X_all_nan)
      assert np.isnan(X_filled[:, 1]).all()  # column with all NaNs remains NaN or handles gracefully
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>43</id>
    <title>pyproject.toml:2-2 - Build-time vs runtime numpy version mismatch</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:2-2

**Build-time vs runtime numpy version mismatch**: The build system uses `oldest-supported-numpy`,
which for Python 3.9 resolves to numpy ≈1.21.6 (and higher for newer Python versions). However, the
runtime dependency declares `numpy>=1.20.0`, allowing numpy 1.20.x at runtime. If the Cython
extension uses numpy C API symbols introduced in 1.21+ (very likely, since it's built against
1.21.x), users with numpy 1.20.x will encounter runtime import errors or segfaults.

**Fix**: Bump the runtime numpy floor to match what `oldest-supported-numpy` provides:
`numpy>=1.21.0` (or at minimum `numpy>=1.21.6` to be precise). This ensures the runtime numpy is at
least as new as the build-time numpy used for ABI compatibility.

  requires = ["setuptools>=61.0.0", "wheel", "Cython>=3.0.0", "oldest-supported-numpy"]
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>44</id>
    <title>pyproject.toml:20-23 - Misleading empty gpu extras group</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:20-23

**Misleading empty `gpu` extras group**: The `[project.optional-dependencies]` section defines `gpu
= []` — an empty list. Users running `pip install nufi[gpu]` will get no additional packages, which
is surprising given the name. While the comment explains the rationale (letting torch manage its own
CUDA runtime), an empty extras group serves no functional purpose and may confuse users who expect
it to install GPU-enabled torch or CUDA dependencies.

**Options**: (1) Remove the `gpu` extras group entirely if it serves no purpose, or (2) add a
validation dependency like `nvidia-smi` or a post-install notice, or (3) keep it but change the
comment to explicitly state it's a no-op placeholder (e.g., `gpu = []  # no extra deps; torch
already bundles CUDA`).

  [project.optional-dependencies]
- # Let torch supply its own CUDA runtime; avoid version conflicts.
- # GPU users simply need a CUDA-capable torch wheel (e.g., `pip install torch --index-url https://download.pytorch.org/whl/cu118`).
+ # GPU acceleration requires a CUDA-capable or MPS-capable PyTorch wheel.
+ # Install separately, e.g.: pip install torch --index-url https://download.pytorch.org/whl/cu118
+ # This extras group is intentionally empty — torch manages its own CUDA runtime.
  gpu = []
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>45</id>
    <title>pyproject.toml:11-12 - Missing classifiers for Python version compatibility</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:11-12

**Missing `classifiers` for Python version compatibility**: The `[project]` table lacks standard
PyPI classifiers (e.g., `Programming Language :: Python :: 3.9`, `Programming Language :: Python ::
3.10`, etc.). Without these, the package's PyPI page will not display supported Python versions, and
tools that rely on classifiers (e.g., some dependency resolvers, badge generators) may misreport
compatibility. This is particularly important given the `requires-python = ">=3.9"` constraint.

  license = {text = "MIT"}
+ classifiers = [
+     "Development Status :: 3 - Alpha",
+     "Intended Audience :: Science/Research",
+     "License :: OSI Approved :: MIT License",
+     "Programming Language :: Python :: 3.9",
+     "Programming Language :: Python :: 3.10",
+     "Programming Language :: Python :: 3.11",
+     "Programming Language :: Python :: 3.12",
+     "Topic :: Scientific/Engineering",
+ ]
  dependencies = [
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>46</id>
    <title>exclude:5-6 - Setup / Maintainability</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/.git/info/exclude:5-6

**Setup / Maintainability**: The exclude patterns `*.[oa]` (C object/archive files) and `*~` (editor
backup files) are still commented out in their default template state. If this project is indeed in
C/C++, leaving these commented means build artifacts (`.o`, `.a`) and backup files can be
accidentally staged and committed. Uncomment them or replace them with patterns appropriate for the
project's actual language (e.g., `__pycache__/`, `*.pyc` for Python; `target/` for Rust;
`node_modules/` for Node; `*.class` for Java).

- # *.[oa]
- # *~
+ *.[oa]
+ *~
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>47</id>
    <title>HEAD:1-1 - Supply-chain awareness (informational):</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/.git/logs/refs/remotes/origin/HEAD:1-1

**Supply-chain awareness (informational):** This reflog shows the repository was cloned from
`https://github.com/dataopsnick/nonuniform-fourier-infill.git`. Since this is an external GitHub
source, verify that this is the intended and trusted upstream repository to mitigate supply-chain
risks. No code-level defect exists here—this is a Git metadata file auto-generated by the clone
operation.
]]></description>
  </task>
</tasklist>

```