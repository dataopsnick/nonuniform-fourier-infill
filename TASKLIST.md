```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>benchmark.py:89-93 - Catching bare Exception is overly broad — it will also ...</title>
    <description><![CDATA[
### Location: tests/benchmark.py:89-93

Catching bare `Exception` is overly broad — it will also catch `KeyboardInterrupt` and `SystemExit`,
preventing the user from aborting the benchmark. Consider narrowing back to the original set
`(ValueError, RuntimeError, ImportError)` plus the new `ValueError` that may now be raised from the
empty-channel check. If you genuinely need a catch-all for robustness in a benchmark script, at
minimum re-raise `KeyboardInterrupt` and `SystemExit`.

-     except Exception as e:
+     except (ValueError, RuntimeError, ImportError) as e:
          import traceback
          print(f"[WARN] NUFI benchmark failed: {e}")
          traceback.print_exc()
          results["NUFI"] = {"Error": f"{type(e).__name__}: {e}"}
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>setup.py:79-79 - This candidate path is incorrect for conda/virtualenv env...</title>
    <description><![CDATA[
### Location: setup.py:79-79

This candidate path is incorrect for conda/virtualenv environments. It expands to
`${sys.prefix}/lib`, so the existence check below (`os.path.join(candidate, "include")`) looks for
`${sys.prefix}/lib/include`. In conda and most virtualenvs, OpenMP headers live at
`${sys.prefix}/include` — not inside a `lib/` subdirectory. As a result, OpenMP will not be
auto-detected on conda setups, silently disabling multi-threading at runtime. The fix is to use
`sys.prefix` directly as the candidate (without appending `"lib"`), so the include check resolves to
`${sys.prefix}/include` and the linker will use `${sys.prefix}/lib`.

-             os.path.join(sys.prefix, "lib"),  # conda / virtualenv
+             sys.prefix,  # conda / virtualenv (include/ and lib/ live directly under prefix)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>setup.py:154-154 - The python_requires here (3.8) contradicts requires-py...</title>
    <description><![CDATA[
### Location: setup.py:154-154

The `python_requires` here (3.8) contradicts `requires-python = ">=3.9"` in pyproject.toml. When
both are present, setuptools merges them, effectively constraining to the stricter of the two — but
the inconsistency is confusing and can lead to incorrect metadata in built wheels (the wheel's
`Requires-Python` may not match expectations). Align this with pyproject.toml by using `">=3.9"` or
remove the duplicate from `setup.py` entirely, since pyproject.toml is the canonical source for PEP
621 metadata.

-     python_requires=">=3.8",
+     python_requires=">=3.9",
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>impute.py:109-112 - Catching generic Exception is too broad. This will sile...</title>
    <description><![CDATA[
### Location: nufi/impute.py:109-112

Catching generic `Exception` is too broad. This will silently suppress critical errors like
`KeyboardInterrupt` (preventing user from stopping execution), `MemoryError` (masking GPU OOM), or
unexpected bugs in the SVD path. The original `RuntimeError` was already broad but targeted at
torch-level errors. Consider catching `(RuntimeError, torch.linalg.LinAlgError, ValueError)`
instead, or at minimum re-raise `KeyboardInterrupt` and `SystemExit`.

-                 except Exception as e:
+                 except (RuntimeError, torch.linalg.LinAlgError, ValueError) as e:
                      import warnings
                      warnings.warn(f"SVD failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
                      continue
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>5</id>
    <title>impute.py:200-201 - New instance attributes self.reconstructed_ and self.c...</title>
    <description><![CDATA[
### Location: nufi/impute.py:200-201

New instance attributes `self.reconstructed_` and `self.coefficients_` store full-length per-column
arrays and can grow unbounded for wide datasets. Consider making this opt-in (e.g., via a
`store_reconstruction` parameter defaulting to `False`) or documenting the memory implications in
the class docstring, since users may not expect `transform()` to accumulate potentially large state.

+         # Note: these dictionaries store per-column full-length arrays and may be large for many-column datasets.
          self.reconstructed_ = {}
          self.coefficients_ = {}
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>6</id>
    <title>torch_kernels.py:152-152 - BUG: valid_idx returned by covariance_compensation in...</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:152-152

BUG: `valid_idx` returned by `covariance_compensation` indexes into the `2*M × 2*M` covariance
matrix (where M = len(X_list)), but the caller in `impute.py` (line 157) treats these as indices
into `valid_cols` which has only M elements. For any M > 0, `len(valid_idx)` >= M (it is exactly 2*M
when no degenerate columns exist), causing an `IndexError` at `valid_cols[idx]` for idx >= M.

Fix: Either (a) return signal-level indices from `covariance_compensation` by mapping
covariance-column indices back to signal indices via `idx // 2` with `np.unique`, or (b) have the
caller convert: `signal_valid = np.unique(valid_idx_comp // 2)`. The latter is simpler:

```python
signal_valid_idx = np.unique(valid_idx_comp // 2)
actual_valid_cols = [valid_cols[idx] for idx in signal_valid_idx]
```

- valid_idx = np.arange(covariance_matrix.shape[0])  # default: all valid
+ # Map covariance-matrix indices (2*M) back to signal indices (M)
+ raw_valid_idx = np.arange(covariance_matrix.shape[0])
+ signal_valid_idx = np.unique(raw_valid_idx // 2)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>7</id>
    <title>torch_kernels.py:176-176 - Correspondingly, update the return statement to return si...</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:176-176

Correspondingly, update the return statement to return signal-level indices so callers receive
indices directly usable with `valid_cols`.

-     return lu, d, perm, valid_idx
+     return lu, d, perm, valid_idx  # returns covariance-matrix indices (length up to 2*M); caller must map to signal space via // 2
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>torch_kernels.py:72-76 - CRASH: compute_ND_NUDFT now raises MemoryError when ...</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:72-76

CRASH: `compute_ND_NUDFT` now raises `MemoryError` when `N > MAX_MEM_N` instead of issuing a
warning. The only internal caller `covariance_compensation` (line 134) does not catch this, and
`impute.py`'s `fit` method (around line 149) has no try/except around the `covariance_compensation`
call. This will cause an unhandled crash for users with large time series.

Consider either: (a) wrapping the `covariance_compensation` call in `fit()` with a try/except that
falls back to no covariance compensation with a warning, or (b) keeping the old warning behavior
with a configurable threshold.

-             raise MemoryError(
-                 f"N={N} exceeds MAX_MEM_N={MAX_MEM_N}; compute_ND_NUDFT would allocate "
-                 f"a ({len(v_timestamps)} × {N}) complex tensor. "
-                 f"Use compute_Fast_ND_NUDFT or reduce N."
+             import warnings
+             warnings.warn(
+                 f"N={N} exceeds MAX_MEM_N={MAX_MEM_N}; compute_ND_NUDFT may consume "
+                 f"excessive memory ({len(v_timestamps)} × {N} complex tensor). "
+                 f"Consider using compute_Fast_ND_NUDFT or reduce N."
              )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>agent.py:60-64 - If the CSV write on line 76 fails, an orphan log entry (e...</title>
    <description><![CDATA[
### Location: nufi/agent.py:60-64

If the CSV write on line 76 fails, an orphan log entry (event: "snapshot_saving") persists with no
corresponding file. The old code had rollback logic (removing the CSV on log failure), but this
version has no cleanup mechanism for the reverse case. Additionally, the event name changed from
"snapshot_saved" to "snapshot_saving", which may break external log consumers (monitoring scripts,
dashboards) that filter on the former event name.

          with self._lock:
-             # Write log entry *before* CSV to enable orphan detection
+             # Write CSV first (if it fails, no log pollution)
+             try:
+                 df.to_csv(filepath, index=True)
+             except Exception as e:
+                 raise TransformationLoggingError(f"Failed to save data snapshot {filepath}: {e}")
+             # Log only after successful write to avoid orphan entries
              log_entry = {
                  "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
-                 "event": "snapshot_saving",
+                 "event": "snapshot_saved",
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>10</id>
    <title>agent.py:230-238 - The warning advises users to normalize timestamps but the...</title>
    <description><![CDATA[
### Location: nufi/agent.py:230-238

The warning advises users to normalize timestamps but the code continues using raw timestamps that
exceed float64 precision (2^53). This can cause catastrophic precision loss in the NUDFT solver:
adjacent timestamps may become indistinguishable, corrupting frequency analysis and imputation
results. Consider automatically subtracting an epoch (e.g., `raw_timestamps - raw_timestamps[0]`)
and recording the offset so downstream results remain interpretable.

-     raw_timestamps = df_copy.index.to_numpy(dtype=np.float64)
-     max_ts = np.max(np.abs(raw_timestamps)) if len(raw_timestamps) > 0 else 0
+     timestamps = df_copy.index.to_numpy(dtype=np.float64)
+     max_ts = np.max(np.abs(timestamps)) if len(timestamps) > 0 else 0
      if max_ts > 2**53:
          import warnings
+         # Subtract epoch to preserve relative precision in float64
+         epoch = timestamps[0] if len(timestamps) > 0 else 0.0
+         timestamps = timestamps - epoch
          warnings.warn(
              f"Timestamps exceed float64 precision (max={max_ts:.1e}). "
-             f"Consider normalizing by subtracting an epoch to preserve relative precision."
+             f"Normalized by subtracting epoch={epoch} to preserve relative precision."
          )
-     timestamps = raw_timestamps
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>agent.py:463-466 - When num_cols == 0, the function returns None implici...</title>
    <description><![CDATA[
### Location: nufi/agent.py:463-466

When `num_cols == 0`, the function returns `None` implicitly. Callers that unpack the return value
(e.g., `fig, axes = plot_diagnostics(...)`) will encounter a `TypeError: cannot unpack non-iterable
NoneType object`. Consider returning an empty figure and axes tuple, or explicitly documenting that
callers must check for a falsy return.

      if num_cols == 0:
          import warnings
          warnings.warn("No columns to plot. Returning empty figure.")
-         return
+         fig, axes = plt.subplots(1, 1, figsize=(8, 4))
+         axes = np.array([[axes]])
+         return fig, axes
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>agent.py:286-287 - The hasattr guards silently fall back to hardcoded defa...</title>
    <description><![CDATA[
### Location: nufi/agent.py:286-287

The `hasattr` guards silently fall back to hardcoded defaults (`1e-4` for alpha, `len(col_data)` for
`n_f`) if the imputer's `alphas_` or `n_frequencies_` attributes are missing. Since these attributes
are always set during `fit()` (see `nufi/impute.py` lines 127-128), their absence indicates a
fitting failure that should be surfaced, not silently masked. Removing `hasattr` would let the
natural `AttributeError` propagate and reveal the root cause.

-         opt_alpha = imputer.alphas_[col_idx] if hasattr(imputer, 'alphas_') and col_idx < len(imputer.alphas_) else 1e-4
-         n_f = imputer.n_frequencies_[col_idx] if hasattr(imputer, 'n_frequencies_') and col_idx < len(imputer.n_frequencies_) else len(col_data)
+         opt_alpha = imputer.alphas_[col_idx] if col_idx < len(imputer.alphas_) else 1e-4
+         n_f = imputer.n_frequencies_[col_idx] if col_idx < len(imputer.n_frequencies_) else len(col_data)
]]></description>
  </task>
</tasklist>

```