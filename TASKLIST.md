```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>test_covariance.py:46-49 - Fragile assertion: Fourier derivative may legitimately ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:46-49

**Fragile assertion: Fourier derivative may legitimately exceed 1.5× linear interpolation's max
derivative.**

Fourier-based imputation can exhibit Gibbs ringing near gap boundaries, producing transient
derivative spikes that exceed those from linear interpolation. The 1.5× margin is arbitrary and may
cause false-positive test failures for valid implementations, especially with the `cg` method or
different gap widths / signal frequencies. Consider removing this cross-method comparison or
replacing it with a direct check against analytic bounds (like the 3× amplitude check above) that
don't depend on another interpolation method's behavior.

-     # Fourier infill should not introduce larger derivative spikes than linear interpolation.
-     # Allow small margin: Fourier should be smoother, but not guaranteed to be strictly <= linear
-     assert np.max(np.abs(dx)) <= 1.5 * np.max(np.abs(lin_dx))
-     assert np.max(np.abs(ddx)) <= 1.5 * np.max(np.abs(lin_ddx))
+     # Fourier infill should produce derivatives bounded by the signal's analytic amplitude.
+     # Avoid comparing against linear interpolation, which is an implementation detail that
+     # can vary across methods and gap configurations.
+     assert np.max(np.abs(dx)) < 5 * amp * omega * dt, (
+         f"First derivative too large: {np.max(np.abs(dx)):.2e} > {5 * amp * omega * dt:.2e}"
+     )
+     assert np.max(np.abs(ddx)) < 5 * amp * omega**2 * dt**2, (
+         f"Second derivative too large: {np.max(np.abs(ddx)):.2e} > {5 * amp * omega**2 * dt**2:.2e}"
+     )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>test_covariance.py:79-80 - Brittle assertion: assumes covariance compensation is always strictly better.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:79-80

**Brittle assertion: assumes covariance compensation is always strictly better.**

With only 50 samples and ~20% missing data, sampling noise alone can cause the uncompensated
covariance to coincidentally be closer to the original. The unconditional `<=` assertion will
hard-fail in such cases, masking whether compensation actually helps on average. Consider using a
statistical check (e.g., repeat over multiple random seeds and assert the mean improvement) or
relaxing this to a softer check with a tolerance margin.

-     # Compensated should be closer to original than uncompensated
-     assert np.linalg.norm(filled_cov - original_cov) <= np.linalg.norm(no_comp_cov - original_cov)
+     # Compensated should be closer to original than uncompensated.
+     # Allow a small tolerance so that sampling noise with only 50 points doesn't
+     # cause spurious failures.
+     assert np.linalg.norm(filled_cov - original_cov) <= 1.05 * np.linalg.norm(no_comp_cov - original_cov), (
+         f"Compensated cov distance {np.linalg.norm(filled_cov - original_cov):.2e} "
+         f"not better than uncompensated {np.linalg.norm(no_comp_cov - original_cov):.2e}"
+     )
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>3</id>
    <title>torch_kernels.py:146-147 - Frequency axis misalignment in covariance_compensation</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:146-147

**Frequency axis misalignment in covariance_compensation**: `compute_ND_NUDFT` is called without
`nyquist_frequency`, so each signal in `X_list` estimates its own Nyquist frequency from its local
sampling intervals. This means different signals produce NUDFT coefficients evaluated on *different
frequency grids* (different `f_k`). The subsequent stacking of real/imag parts and covariance
computation then mixes incompatible frequency bins — the resulting covariance matrix is
mathematically invalid because column `j` of signal A and column `j` of signal B correspond to
different frequencies.

**Fix**: Pass a common `nyquist_frequency` computed from all signals (e.g., the most conservative
estimate across the batch) or pass `nyquist_frequency` as a parameter to `covariance_compensation`.

-     # Step 1: Compute NUDFT results
-     X_k_result = compute_ND_NUDFT(X_list, device=dev)
+     # Step 1: Compute NUDFT results with a shared Nyquist frequency
+     # Estimate a common Nyquist across all signals to ensure aligned frequency bins
+     all_diffs = []
+     for X in X_list:
+         ts = np.array(X.timestamps, dtype=np.float64)
+         v_ts = ts[~np.isnan(ts)]
+         if len(v_ts) > 1:
+             sorted_ts = np.sort(v_ts)
+             diffs = np.diff(sorted_ts)
+             all_diffs.extend(diffs[diffs > 0].tolist())
+     if all_diffs:
+         common_nyquist = 0.5 / max(np.median(all_diffs), 1e-12)
+     else:
+         common_nyquist = 1.0
+     X_k_result = compute_ND_NUDFT(X_list, device=dev, nyquist_frequency=common_nyquist)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>4</id>
    <title>test_covariance.py:51-67 - Incomplete coverage: test_covariance_preservation only te...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_covariance.py:51-67

Incomplete coverage: test_covariance_preservation only tests method='direct'. The cg method may
behave differently for covariance compensation and should also be exercised. Additionally, the
imputer_no_comp instantiation on line ... would also need to use the parameterized method value once
the test is parametrized; otherwise the parametrized test would always compare against
method='direct' for the uncompensated case, defeating the purpose. Add
@pytest.mark.parametrize('method', ['direct', 'cg']) and use the parameterized method in both
imputer instantiations.

- def test_covariance_preservation():
+ @pytest.mark.parametrize("method", ["direct", "cg"])
+ def test_covariance_preservation(method):
      # Verify that multi-signal covariance is maintained after imputation
      t = np.linspace(0, 10, 50)
      s1 = np.sin(t)
      s2 = np.cos(t)
      
      original_cov = np.cov(s1, s2)
      
      # Introduce NaNs
      s1_nan = s1.copy()
      s2_nan = s2.copy()
      s1_nan[10:20] = np.nan
      s2_nan[30:40] = np.nan
      
      X = np.stack([s1_nan, s2_nan], axis=1)
      
-     imputer = NufiImputer(method='direct', covariance_compensation=True)
+     imputer = NufiImputer(method=method, covariance_compensation=True)
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>5</id>
    <title>torch_kernels.py:83-84 - NUDFT sign convention mismatch between forward and inverse transforms</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:83-84

**NUDFT sign convention mismatch between forward and inverse transforms**: `compute_ND_NUDFT` uses
the analysis (forward) convention `exp(-2πi * t_n * f_k)`, while `solve_tikhonov_nudft` uses the
synthesis (inverse) convention `exp(+2πi * t_n * f_k)`. If a downstream caller obtains coefficients
via `compute_ND_NUDFT` and then reconstructs with `solve_tikhonov_nudft`, the sign inversion will
silently corrupt the phase of the result. Since both functions live in the same module, this is a
latent correctness bug.

**Fix**: Either unify the conventions across the module, or add a prominent cross-reference warning
in both docstrings (e.g., "If you obtained coefficients from `compute_ND_NUDFT`, you must conjugate
them before passing to this function").

          # Forward NUDFT: A[n,k] = exp(-2πi * t_n * f_k)  (analysis convention)
+         # WARNING: solve_tikhonov_nudft uses the synthesis convention exp(+2πi*t_n*f_k).
+         # Conjugate coefficients before passing between these functions to avoid phase errors.
          exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>6</id>
    <title>torch_kernels.py:324-329 - Memory guard missing for augmented system in direct solver</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:324-329

**Memory guard missing for augmented system in direct solver**: The guard at line 277 (`N * M >
MAX_ELEMENTS`) only checks the size of `A` (shape `N×M`). However, the direct path then constructs
an augmented matrix of shape `(N+M)×M`, which holds `N*M + M²` elements — roughly double the memory
when `N ≈ M`. The guard can pass but the augmented allocation can still OOM.

**Fix**: Extend the guard to account for the augmented matrix size, e.g., `(N + M) * M >
MAX_ELEMENTS`.

      MAX_ELEMENTS = 50_000_000  # ~800 MB for complex128
-     if N * M > MAX_ELEMENTS and solver != 'cg':
+     if (N + M) * M > MAX_ELEMENTS and solver != 'cg':
          raise ValueError(
-             f"Matrix A shape ({N},{M}) has {N*M} elements; exceeds memory safety limit. "
+             f"Augmented matrix shape ({N+M},{M}) has {(N+M)*M} elements; exceeds memory safety limit. "
              f"Use solver='cg' to avoid materializing the full matrix."
          )
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>7</id>
    <title>torch_kernels.py:191-202 - Weak regularization may still lead to LDL failure with no fallback</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:191-202

**Weak regularization may still lead to LDL failure with no fallback**: The regularization epsilon
is `max(1e-10 * max(diag_mean, 0.0), 1e-10)`. When `diag_mean` is very close to zero, the effective
regularization is only `1e-10`, which may be insufficient for a nearly singular covariance matrix.
If LDL still fails, a `ValueError` is raised with no recovery path — the caller has no way to retry
with a stronger epsilon.

**Fix**: Either use a more robust scaling (e.g., `eps = max(1e-10 * max(diag_mean, 0.0), 1e-8)`) or
wrap the LDL call in a retry loop that progressively increases epsilon on failure.

      # Regularize to ensure positive-definiteness for LDL^T
      # Scale epsilon relative to the matrix magnitude for robustness
      diag_mean = np.mean(np.diag(covariance_matrix))
      eps = max(1e-10 * max(diag_mean, 0.0), 1e-10)
-     covariance_matrix = covariance_matrix + eps * np.eye(covariance_matrix.shape[0])
- 
-     # Step 4: Perform LDL^T decomposition
-     # LDL^T factorizes A = P * L * D * L^T
+     # Retry with progressively larger regularization on failure
+     max_retries = 5
+     for attempt in range(max_retries):
+         cov_reg = covariance_matrix + eps * np.eye(covariance_matrix.shape[0])
      try:
-         lu, d, perm = scipy.linalg.ldl(covariance_matrix)
+             lu, d, perm = scipy.linalg.ldl(cov_reg)
+             break
      except Exception:
-         raise ValueError("LDL decomposition failed; covariance matrix may be singular.")
+             if attempt == max_retries - 1:
+                 raise ValueError("LDL decomposition failed even after regularization; covariance matrix may be singular.")
+             eps *= 10.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>8</id>
    <title>setup.py:9-11 - Bug: parse_version incorrectly handles pre-release identifiers.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:9-11

**Bug: `parse_version` incorrectly handles pre-release identifiers.**

For input like `"3.0.0a1"`, the function strips all non-digit/non-dot characters producing
`"3.0.0.1"`, which parses to `(3, 0, 0, 1)`. Tuple comparison `(3, 0, 0, 1) < (3, 0, 0)` evaluates
to `False`, so the pre-release incorrectly passes the `>= 3.0.0` gate. Similarly, `"3.0.0rc1"` →
`(3,0,0,1)`, `"3.0.0.dev0"` → `(3,0,0,0)`, and `"1.20.0rc1"` → `(1,20,0,1)` all sneak through. This
means incompatible pre-release Cython or NumPy versions could be accepted at build time, leading to
confusing downstream errors.

**Suggestion:** Since `packaging` is already listed in `install_requires`, use
`packaging.version.parse` for standards-compliant PEP 440 version comparison, e.g.:
```python
from packaging.version import parse as parse_version
```
Or, at minimum, strip pre-release/post-release/dev segments before converting to a numeric tuple.

- def parse_version(v_str):
-     clean_v = "".join(c if c.isdigit() or c == "." else "" for c in v_str.split("-")[0].split("+")[0])
-     return tuple(map(int, [p for p in clean_v.split(".") if p]))
+ from packaging.version import parse as parse_version
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>9</id>
    <title>setup.py:89-96 - Bug: macOS OpenMP detection only checks file existence, not functional linking.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/setup.py:89-96

**Bug: macOS OpenMP detection only checks file existence, not functional linking.**

Unlike the Linux branch (which performs an actual compile-and-link test with `subprocess.run`), the
macOS branch only verifies that `omp.h` and `libomp.dylib` exist. If the found libomp is for a
mismatched architecture (e.g., arm64 library on an x86_64 Python) or is otherwise incompatible with
the compiler, the file-existence check passes silently and the build will fail later with a cryptic
linker error instead of a clear warning + graceful fallback.

**Suggestion:** Perform a compile-and-link test similar to the Linux branch, using the discovered
`-Xpreprocessor -fopenmp` flags and libomp path. On failure, fall back to no OpenMP with a clear
warning.

          if libomp_path:
+             # Verify the discovered libomp actually links
+             tmpdir = tempfile.mkdtemp()
+             try:
+                 test_file = os.path.join(tmpdir, "test.c")
+                 with open(test_file, "w") as f:
+                     f.write("#include <omp.h>\nint main(void) { return omp_get_num_threads(); }\n")
+                 cc = os.environ.get("CC") or sysconfig.get_config_var("CC") or "cc"
+                 cmd = [
+                     cc, "-Xpreprocessor", "-fopenmp",
+                     "-I" + os.path.join(libomp_path, "include"),
+                     "-L" + os.path.join(libomp_path, "lib"),
+                     "-lomp",
+                     test_file, "-o", os.path.join(tmpdir, "test"),
+                 ]
+                 res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
+                 if res.returncode != 0:
+                     raise RuntimeError(f"Link test failed: {res.stderr.decode().strip()}")
+             except Exception as e:
+                 warnings.warn(f"OpenMP link test failed: {e}. Falling back to no OpenMP.")
+                 ext_compiler_args = []
+                 ext_linker_args = []
+                 # Remove the previously appended include dir
+                 ext_include_dirs.pop()
+             finally:
+                 shutil.rmtree(tmpdir, ignore_errors=True)
+             else:
              ext_compiler_args = ["-Xpreprocessor", "-fopenmp"]
              ext_linker_args = [
                  "-L" + os.path.join(libomp_path, "lib"),
                  "-Wl,-rpath," + os.path.join(libomp_path, "lib"),
                  "-lomp",
              ]
              ext_include_dirs.append(os.path.join(libomp_path, "include"))
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>10</id>
    <title>torch_kernels.py:47-49 - Silently diverging Nyquist estimates across batch calls</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:47-49

**Silently diverging Nyquist estimates across batch calls**: When `nyquist_frequency` is `None`,
each signal independently estimates its own Nyquist from local sampling intervals. In multi-signal
workflows (like `covariance_compensation`), this causes silently mismatched frequency grids. Even
when used alone, the user gets no indication that per-signal estimates may differ.

**Fix**: At a minimum, emit a `warnings.warn` when per-signal Nyquist estimates differ across a
batch, or require the caller to provide `nyquist_frequency` explicitly in multi-signal contexts.

          # Use caller-provided Nyquist if available, otherwise estimate from sampling
          if nyquist_frequency is None:
+             import warnings
+             warnings.warn(
+                 "nyquist_frequency not provided; estimating per-signal Nyquist. "
+                 "This may produce inconsistent frequency grids across signals. "
+                 "Pass an explicit nyquist_frequency for multi-signal workflows."
+             )
              if len(v_timestamps) > 1:
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>11</id>
    <title>benchmark.py:86-90 - Exception handling is too narrow across multiple benchmar...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:86-90

Exception handling is too narrow across multiple benchmark methods (NUFI, Cubic Spline, MICE,
Gaussian Process). Only ValueError, RuntimeError, and ImportError are caught. Methods can raise
TypeError, AttributeError, KeyError, or numpy.linalg.LinAlgError (e.g., from GP fitting or
covariance computation), which would crash the entire benchmark rather than recording the error for
that method. Consider catching Exception (or at least broadening to include TypeError and
LinAlgError) so the benchmark is resilient and reports failures per-method.

-     except (ValueError, RuntimeError, ImportError) as e:
+     except Exception as e:
          import traceback
          print(f"[WARN] NUFI benchmark failed: {e}")
          traceback.print_exc()
          results["NUFI"] = {"Error": f"{type(e).__name__}: {e}"}
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>12</id>
    <title>benchmark.py:166-169 - When a channel has n_valid == 0, the entire column is f...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:166-169

When a channel has `n_valid == 0`, the entire column is filled with `np.nan`. This propagates
silently into the RMSE and covariance computations: `NaN ** 2` produces `NaN`, `np.mean(NaN)`
produces `NaN`, and `np.linalg.norm` of a NaN-containing matrix returns `NaN` without raising an
exception. The result is a misleading "successful" benchmark entry with NaN metrics. Consider
detecting NaN in the results and reporting an error status, or computing metrics only on
rows/columns where imputation actually produced finite values.

                      if n_valid < 2:
                          # Not enough observations to fit a GP; fill with NaN to avoid biased scores
                          gp_infilled_data[:, c] = np.nan if n_valid == 0 else np.nanmean(col_data)
                          continue
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>13</id>
    <title>benchmark.py:76-79 - np.linalg.norm returns NaN (without raising an exception)...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:76-79

np.linalg.norm returns NaN (without raising an exception) when the input matrix contains NaN values.
This applies to multiple benchmark methods (NUFI, Cubic Spline, MICE, Gaussian Process). The
try/except only catches ValueError and LinAlgError, so NaN covariance errors go undetected and
produce misleading benchmark results. Consider adding an explicit np.isnan() guard after the norm
call, or checking the covariance matrix for NaN before computing the norm.

          try:
-             nufi_cov_err = np.linalg.norm(true_cov - nufi_infilled.cov().to_numpy(), ord='fro')
+             cov_diff = true_cov - nufi_infilled.cov().to_numpy()
+             if np.any(np.isnan(cov_diff)):
+                 nufi_cov_err = float('nan')
+             else:
+                 nufi_cov_err = np.linalg.norm(cov_diff, ord='fro')
          except (ValueError, np.linalg.LinAlgError):
              nufi_cov_err = float('nan')
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>14</id>
    <title>benchmark.py:157-159 - The GP skip condition uses max() valid points across al...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/benchmark.py:157-159

The GP skip condition uses `max()` valid points across all channels. If any single channel exceeds
`gp_max_valid`, the entire GP method is skipped — even if other channels are sparse and could be
benchmarked successfully. Consider skipping only channels that exceed the threshold while still
fitting GPs on the remaining channels, or at least logging which channel triggered the skip.

              n_valid_max = max((~np.isnan(df_masked.to_numpy()[:, c])).sum() for c in range(n_channels))
              if n_valid_max > gp_max_valid:
                  results["Gaussian Process"] = {"Status": f"Skipped: too many valid points ({n_valid_max}) for O(n³) GP"}
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>15</id>
    <title>torch_kernels.py:336-338 - Unvalidated solver parameter in solve_tikhonov_nudft</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:336-338

**Unvalidated `solver` parameter in `solve_tikhonov_nudft`**: The `solver` parameter defaults to
`'direct'` but there is no validation of its value. If a caller passes `solver='svd'` or a typo like
`solver='direc'`, execution silently falls into the `else` branch (the direct augmented-system
solver) rather than raising an informative error. This can lead to unexpected behavior and memory
issues.

**Fix**: Validate `solver` at the top of the function:
```python
if solver not in ('direct', 'cg'):
    raise ValueError(f"Unknown solver '{solver}'. Supported: 'direct', 'cg'.")
```

+     if solver not in ('direct', 'cg'):
+         raise ValueError(f"Unknown solver '{solver}'. Supported solvers: 'direct', 'cg'.")
      if solver == 'cg':
          F = solve_cg(A, b, alpha, max_iter=max_iter, tol=tol)
      else:
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>16</id>
    <title>test_agent.py:145-163 - Missing test for invalid version ID in revert_to_version....</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:145-163

Missing test for invalid version ID in revert_to_version. The current test
(test_tracker_version_reversion) only covers the happy path. revert_to_version may receive
non-existent IDs, malformed IDs, or IDs from a different tracker instance, which could cause
unhandled exceptions, silent corruption, or returning stale data. Add test cases for: (a)
non-existent version ID, (b) empty string ID, (c) integer/non-string ID type.

      def test_tracker_version_reversion(self):
          tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
          
          # Save a snapshot of the original dataframe
          df_orig = self.df.copy()
          ver_id = tracker.save_snapshot(df_orig, "step_1_orig")
          
          # Mutate the dataframe
          df_mutated = df_orig.copy()
          df_mutated["signal"] = 999.0
          
          # Verify reversion returns exactly the original data
          df_reverted = tracker.revert_to_version(ver_id)
          pd.testing.assert_frame_equal(df_orig, df_reverted)
          # Guard against false positive: ensure reverted is not the mutated version
          self.assertFalse(
              df_reverted["signal"].equals(df_mutated["signal"]),
              "revert_to_version returned the mutated dataframe instead of the original snapshot"
          )
+ 
+     def test_tracker_revert_to_version_invalid_id(self):
+         """Edge case: revert_to_version with non-existent ID should raise an error."""
+         tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
+         with self.assertRaises((KeyError, FileNotFoundError, ValueError)):
+             tracker.revert_to_version("nonexistent_id_12345")
+ 
+     def test_tracker_revert_to_version_malformed_id(self):
+         """Edge case: revert_to_version with empty or wrong-type IDs."""
+         tracker = TransformationTracker(log_path=self.test_log, history_dir=self.test_history)
+         for bad_id in ["", 42, None]:
+             with self.subTest(bad_id=bad_id):
+                 with self.assertRaises((KeyError, FileNotFoundError, TypeError, ValueError)):
+                     tracker.revert_to_version(bad_id)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>17</id>
    <title>test_agent.py:72-83 - All-NaN test does not verify column integrity of non-NaN ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:72-83

All-NaN test does not verify column integrity of non-NaN columns (e.g., timestamp). The test only
asserts that signal stays NaN, but if impute_dataframe accidentally drops, reorders, or corrupts the
timestamp column, this would go undetected. Add assertions confirming timestamp column is preserved
unchanged.

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
          self.assertTrue(result_df["signal"].isna().all())
+         # Verify timestamp column is preserved unchanged
+         np.testing.assert_array_equal(
+             result_df["timestamp"].values,
+             all_nan_df["timestamp"].values
+         )
          self.assertIn("signal", diagnostics)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>18</id>
    <title>test_agent.py:165-197 - plot_diagnostics test only uses well-formed diagnostics f...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:165-197

plot_diagnostics test only uses well-formed diagnostics from impute_dataframe. No test covers
malformed diagnostics dict (e.g., missing required keys like 'snr_db' or 'spectral_entropy', wrong
types, empty dict). In production, hand-crafted or externally sourced diagnostics could cause
unhandled KeyError or TypeError inside plot_diagnostics. Add tests passing incomplete, empty, or
mis-typed diagnostics dicts.

      def test_agent_plot_diagnostics(self):
          # Ensure non-interactive backend for headless CI environments
          import matplotlib
          matplotlib.use('Agg')
          
          # Run infilling
          infilled_df, diagnostics = impute_dataframe(
              self.df,
              time_col="timestamp",
              log_path=self.test_log,
              history_dir=self.test_history
          )
          
          # Test plot rendering with show_plot=False to avoid blocking tests
          save_img = os.path.join(self._tmpdir, "test_diagnostics_plot.png")
          if os.path.exists(save_img):
              os.remove(save_img)
              
          try:
              plot_diagnostics(
                  original_df=self.df,
                  infilled_df=infilled_df,
                  diagnostics=diagnostics,
                  time_col="timestamp",
                  save_path=save_img,
                  show_plot=False
              )
              
              # Verify the plot image was successfully created on disk
              self.assertTrue(os.path.exists(save_img))
+         finally:
+             if os.path.exists(save_img):
+                 os.remove(save_img)
+ 
+     def test_plot_diagnostics_bad_input(self):
+         """Edge case: plot_diagnostics with malformed diagnostics dict."""
+         import matplotlib
+         matplotlib.use('Agg')
+         clean_df = pd.DataFrame({"timestamp": [1.0, 2.0], "signal": [10.0, 20.0]})
+         save_img = os.path.join(self._tmpdir, "bad_diag_plot.png")
+         bad_diagnostics_cases = [
+             {},                                    # empty dict
+             {"signal": {}},                        # missing all expected keys
+             {"signal": {"snr_db": "not_a_number"}}, # wrong type
+             {"signal": None},                      # None column diag
+         ]
+         for bad_diag in bad_diagnostics_cases:
+             with self.subTest(bad_diag=bad_diag):
+                 try:
+                     plot_diagnostics(
+                         original_df=clean_df,
+                         infilled_df=clean_df,
+                         diagnostics=bad_diag,
+                         time_col="timestamp",
+                         save_path=save_img,
+                         show_plot=False
+                     )
+                 except (KeyError, TypeError, ValueError):
+                     pass  # expected for truly malformed input
          finally:
              if os.path.exists(save_img):
                  os.remove(save_img)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>19</id>
    <title>test_agent.py:166-168 - No guard for matplotlib availability. The test imports ma...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_agent.py:166-168

No guard for matplotlib availability. The test imports matplotlib inside the method body (line 157)
without a try/except. If matplotlib is not installed in the test environment, the test will fail
with ImportError rather than being skipped. Use unittest.skipIf or a try/except that calls
self.skipTest to gracefully handle missing optional dependencies.

          # Ensure non-interactive backend for headless CI environments
+         try:
          import matplotlib
          matplotlib.use('Agg')
+         except ImportError:
+             self.skipTest("matplotlib not available")
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>20</id>
    <title>torch_kernels.py:124-129 - Duplicate timestamps cause undefined behavior in np.interp</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/kernels/torch_kernels.py:124-129

**Duplicate timestamps cause undefined behavior in `np.interp`**: The check
`np.all(np.diff(v_timestamps) >= 0)` permits zero differences (duplicate timestamps). NumPy's
`np.interp` explicitly states that behavior is undefined for duplicate x-values — it may silently
produce incorrect interpolated values.

**Fix**: Filter out duplicate timestamps before interpolation, e.g., by keeping the first or
averaging the data at coincident points.
```python
# Remove duplicates (keep first occurrence)
unique_mask = np.concatenate(([True], np.diff(v_timestamps) > 0))
v_timestamps = v_timestamps[unique_mask]
v_data = v_data[unique_mask]
```

-         # Ensure timestamps are sorted for np.interp (requires monotonic increasing)
+         # Ensure timestamps are sorted for np.interp (requires strictly increasing)
          if not np.all(np.diff(v_timestamps) >= 0):
              sort_idx = np.argsort(v_timestamps)
              v_timestamps = v_timestamps[sort_idx]
              v_data = v_data[sort_idx]
+         # Remove duplicate timestamps (np.interp behavior is undefined for non-unique x)
+         unique_mask = np.concatenate(([True], np.diff(v_timestamps) > 0))
+         v_timestamps = v_timestamps[unique_mask]
+         v_data = v_data[unique_mask]
          uniform_data = np.interp(uniform_grid, v_timestamps, v_data)
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>21</id>
    <title>wrappers.py:228-228 - Bug: imputer.clone() may not exist.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:228-228

**Bug: `imputer.clone()` may not exist.** The code unconditionally calls `imputer.clone()` on line
159 (since `imputer` is always non-None at that point due to the outer-scope initialization on line
121). If `NufiImputer` does not expose a `clone()` method, this will raise `AttributeError` at
runtime, causing a hard crash for every group. Consider either requiring `clone()` in the public API
contract, catching the error with a clear message, or using `copy.deepcopy(imputer)` /
`NufiImputer(**imputer.get_params())` as a fallback.

Additionally, the `if imputer is None` branch on line 159 is dead code — `imputer` is always
initialized before `imputer_apply` is defined, so the `NufiImputer()` fallback here is unreachable.

-         group_imputer = NufiImputer() if imputer is None else imputer.clone()
+         try:
+             group_imputer = imputer.clone()
+         except AttributeError:
+             raise AttributeError(
+                 "NufiImputer does not implement clone(). "
+                 "Provide a separate imputer instance per entity."
+             )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>22</id>
    <title>wrappers.py:212-216 - Bug: NaT values in datetime64 timestamps are not detected.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:212-216

**Bug: NaT values in datetime64 timestamps are not detected.** The monotonicity check computes
`np.diff(timestamps.view('int64'))` for datetime64 arrays, but when `timestamps` contains NaT
values, `np.diff` returns NaT, and `np.any(diffs <= 0)` is `False` (NaT comparisons return `False`).
This means groups with NaT timestamps will silently pass validation, and downstream conversion to
`float64` will fail or produce incorrect imputations.

Add an explicit check for NaT before the diff:
```python
if np.issubdtype(timestamps.dtype, np.datetime64) and np.any(pd.isna(timestamps)):
    raise ValueError("Timestamp level contains NaT values.")
```

              if np.issubdtype(timestamps.dtype, np.datetime64):
+                 if pd.isna(timestamps).any():
+                     raise ValueError("Timestamp level contains NaT values.")
                  diffs = np.diff(timestamps.view('int64'))
              else:
+                 if np.isnan(timestamps.astype(np.float64)).any():
+                     raise ValueError("Timestamp level contains NaN values.")
                  diffs = np.diff(timestamps.astype(np.float64))
              if np.any(diffs <= 0):
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>23</id>
    <title>wrappers.py:101-105 - Bug: infill_dataframe is missing strict monotonicity validation for timestamps.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:101-105

**Bug: `infill_dataframe` is missing strict monotonicity validation for timestamps.** The MultiIndex
wrapper (`infill_multiindex_dataframe`) validates that timestamps are strictly increasing and checks
float64 convertibility (lines 136–151), but the single-index wrapper (`infill_dataframe`) performs
no such checks. Passing non-monotonic or non-convertible timestamps (e.g., string dates, NaT,
duplicate times) will silently produce incorrect frequency estimates or cryptic downstream errors
from the imputer.

Add similar validation after sorting (or before fit_transform) for parity.

          else:
              pd_df = pd_df.set_index(time_col)
              pd_df.index.name = None
+ 
+     # Validate timestamp integrity (parity with multiindex wrapper)
+     timestamps = pd_df.index.values
+     try:
+         np.array(timestamps, dtype=np.float64)
+     except (TypeError, ValueError):
+         raise TypeError(
+             f"Index dtype {pd_df.index.dtype} cannot be converted to float64. "
+             "Use a numeric or datetime64 index."
+         )
+     if len(timestamps) > 1:
+         if np.issubdtype(timestamps.dtype, np.datetime64):
+             if pd.isna(timestamps).any():
+                 raise ValueError("Index contains NaT values.")
+             diffs = np.diff(timestamps.view('int64'))
+         else:
+             diffs = np.diff(timestamps.astype(np.float64))
+         if np.any(diffs <= 0):
+             raise ValueError(
+                 "Timestamps must be strictly increasing; "
+                 "found non-positive difference."
+             )
          
      infilled_pd = imputer.fit_transform(pd_df)
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>24</id>
    <title>wrappers.py:84-97 - Bug: keep_time_col=True produces inconsistent index vs. column after imputation.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:84-97

**Bug: `keep_time_col=True` produces inconsistent index vs. column after imputation.** When
`keep_time_col=True`, the time column is duplicated as both the DataFrame index and a feature column
before `fit_transform`. The imputer may modify the time column values during imputation (e.g.,
filling gaps), but the index is left untouched. The result is a DataFrame where the `time_col`
column has imputed values but the index retains the original timestamps — they will disagree. Users
relying on either the column or the index for downstream work will get inconsistent results.

Consider: (a) after `fit_transform`, overwrite the index with the (now possibly imputed) time column
values, or (b) exclude the time column from imputation features entirely (pass only non-time columns
to the imputer).

          if keep_time_col:
              import warnings
              warnings.warn(
                  "keep_time_col=True duplicates timestamps as both index and feature. "
                  "Timestamp magnitudes (e.g., Unix nanoseconds) may dominate covariance "
                  "estimation and produce biased imputations for other columns. "
                  "Consider normalizing timestamps or using keep_time_col=False.",
                  UserWarning
              )
-             time_values = pd_df[time_col].copy()
              col_pos = pd_df.columns.get_loc(time_col)
              pd_df = pd_df.set_index(time_col)
-             pd_df.index.name = None  # avoid name collision with the column
-             pd_df.insert(col_pos, time_col, time_values)
+             pd_df.index.name = None
+             pd_df.insert(col_pos, time_col, pd_df.index.to_numpy())
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>25</id>
    <title>wrappers.py:262-266 - Bug: Missing MultiIndex integrity check after group concatenation.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/wrappers.py:262-266

**Bug: Missing MultiIndex integrity check after group concatenation.** Only total row count is
validated (line 184), but duplicate or overlapping MultiIndex entries can be silently merged by
`pd.concat`. For example, if two entity groups produce the same index labels, the concatenated
result may have fewer rows than expected without raising an error. Add a check that the concatenated
index is unique and matches the original index structure.

      if len(infilled_pd) != len(pd_df):
          raise ValueError(
              f"Concatenated result has {len(infilled_pd)} rows, "
              f"expected {len(pd_df)}. Group-level indices may overlap or be non-unique."
+         )
+     if not infilled_pd.index.equals(pd_df.index):
+         raise ValueError(
+             "Concatenated result index differs from input index. "
+             "Group-level indices may overlap or be non-unique."
          )
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>26</id>
    <title>impute.py:154-157 - Fallback n_f can exceed N_val for small datasets, causing underdetermined systems.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:154-157

**Fallback n_f can exceed N_val for small datasets, causing underdetermined systems.**

When all GCV candidates fail SVD (e.g., N_val is very small), the fallback `best_n_freq = max(5,
min(N_val - 1, N_val // 2))` produces:
- N_val=1 → n_f=5 (5 unknowns, 1 equation)
- N_val=2 → n_f=5 (5 unknowns, 2 equations)
- N_val=3 → n_f=5 (5 unknowns, 3 equations)

These are severely underdetermined, and the NUDFT solver in `transform()` will likely fail or
produce meaningless results. The fallback should ensure `n_f <= N_val` (or at most `N_val` with a
safe upper bound like `min(5, N_val)`).

              if best_gcv == float('inf'):
                  import warnings
-                 best_n_freq = max(5, min(N_val - 1, N_val // 2))  # ensure n_f < N_val to avoid underdetermined system
+                 best_n_freq = max(1, min(5, N_val - 1))  # ensure n_f < N_val to avoid underdetermined system
                  best_alpha = 1.0
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>27</id>
    <title>impute.py:290-305 - Stochastic imputation with near-zero cov_scale causes catastrophic noise amplification.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:290-305

**Stochastic imputation with near-zero cov_scale causes catastrophic noise amplification.**

When `cov_scale` is extremely small but non-zero (e.g., `1e-16` from a near-zero diagonal in `d_`),
the stochastic branch computes:
```
reconstructed_compensated = reconstructed_np * 1e-16  # ≈ 0
imputed_vals = (0 + noise) / 1e-16                    # noise blown up by 1e16
```
This destroys the signal and fills NaNs with extreme values. The `cov_scale > 0` guard only protects
against exact zero, not near-zero. Consider using a minimum threshold (e.g., `1e-8`) below which
`cov_scale` is treated as zero, or skipping covariance compensation when `d_` eigenvalues are too
small.

              # Fill only the NaNs
              nan_mask = np.isnan(X_data[:, col_idx])
              if np.any(nan_mask):
                  if stochastic:
                      obs_mask = ~nan_mask
                      if np.any(obs_mask):
                          residual = (X_data[obs_mask, col_idx] * cov_scale if cov_scale > 0 else X_data[obs_mask, col_idx]) - reconstructed_compensated[obs_mask]
                          residual_std = np.std(residual) if len(residual) > 1 else 0.1
                          if np.isnan(residual_std) or residual_std == 0:
                              residual_std = 0.1
                      else:
                          residual_std = 0.1
                          
                      noise = rng.normal(0, stochastic_scale * residual_std, size=nan_mask.sum())
-                     # Convert back to original scale for output
-                     imputed_vals = (reconstructed_compensated[nan_mask] + noise) / cov_scale if cov_scale > 0 else reconstructed_compensated[nan_mask] + noise
+                     # Convert back to original scale for output; guard against near-zero cov_scale
+                     if cov_scale > 1e-8:
+                         imputed_vals = (reconstructed_compensated[nan_mask] + noise) / cov_scale
+                     else:
+                         imputed_vals = reconstructed_np[nan_mask] + noise
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>28</id>
    <title>impute.py:196-206 - Covariance compensation permutation mapping may be misa...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:196-206

**Covariance compensation permutation mapping may be misaligned with `d_small`/`lu_small`
indexing.**

`d_small` and `lu_small` are indexed by `i` and `j` in the original (pre-permutation) order of the
small matrix, but the code maps them to `actual_valid_cols[perm_small[i]]` — the *permuted* column
positions. If `covariance_compensation()` returns `d_small`/`lu_small` already in the permuted
order, this is correct; if they are in the original order, the diagonal scaling and off-diagonal
decorrelation are applied to the wrong columns. The internal comment acknowledges this ambiguity.
This should be clarified or the mapping should be verified against the `covariance_compensation`
contract.

-                 # Map small matrices back to full size using perm_small mapping
+                 # Map small matrices back to full size.
+                 # NOTE: d_small[i] and lu_small[i,j] correspond to the *original* (pre-permutation)
+                 # index i in the small matrix. To map to full column indices, we must apply
+                 # the inverse permutation so that the diagonal entry for original position i
+                 # lands at the correct full column. Verify covariance_compensation() ordering.
                  n_small = len(perm_small)
+                 inv_perm = np.argsort(perm_small)
                  for i in range(n_small):
-                     full_i = actual_valid_cols[perm_small[i]]
-                     self.d_[full_i, full_i] = d_small[i]
+                     full_i = actual_valid_cols[i]
+                     self.d_[full_i, full_i] = d_small[inv_perm[i]]
                  
                  for i in range(n_small):
-                     full_i = actual_valid_cols[perm_small[i]]
+                     full_i = actual_valid_cols[i]
                      for j in range(n_small):
-                         full_j = actual_valid_cols[perm_small[j]]
-                         self.lu_[full_i, full_j] = lu_small[i, j]
+                         full_j = actual_valid_cols[j]
+                         self.lu_[full_i, full_j] = lu_small[inv_perm[i], inv_perm[j]]
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>29</id>
    <title>impute.py:31-34 - lu_ (full LDL^T) is computed but never used in trans...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/impute.py:31-34

**`lu_` (full LDL^T) is computed but never used in `transform()` — only `d_` (diagonal) is
applied.**

The class docstring itself notes: "Currently only d_ (diagonal scaling) is applied in transform().
Consider implementing full LDL^T application for proper multi-signal covariance compensation." This
means the off-diagonal covariance structure between signals is ignored during imputation, which
defeats the purpose of `covariance_compensation=True` for multi-signal datasets. The diagonal-only
scaling may also be inconsistent with how `d_` was derived (as part of LDL^T, not as independent
per-signal variances).

-         # Note: lu_ and perm_ are stored for potential downstream use in full
-         # covariance-aware reconstruction. Currently only d_ (diagonal scaling)
-         # is applied in transform(). Consider implementing full LDL^T application
-         # for proper multi-signal covariance compensation.
+         # TODO: Apply full LDL^T in transform() for proper multi-signal covariance
+         # compensation. Currently only d_ (diagonal scaling) is applied, which
+         # ignores off-diagonal correlations captured by lu_ and perm_.
+         # In transform(), use: L = lu_ - np.eye(n_cols) + np.eye(n_cols) (extract unit lower)
+         # Then apply: x_compensated = P @ L @ sqrt(D) @ x (or similar).
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>30</id>
    <title>test_imputer.py:166-168 - Flaky assertion: stochastic non-determinism check may fail by coincidence.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:166-168

**Flaky assertion: stochastic non-determinism check may fail by coincidence.**

Two calls to `transform(stochastic=True)` without a fixed `random_state` are asserted to differ by
`>1e-12`. While extremely unlikely, a random collision is theoretically possible and would cause a
flaky CI failure with a misleading assertion message. Since the very next block already tests
reproducibility with `random_state=42`, consider using seeded and deliberately different random
states (e.g., 42 vs 99) to guarantee divergence, or relax this assertion to check only that the
results are not bitwise-equal without requiring both positions to differ simultaneously.

The assertion error message also says "at least one pair was identical" but the `and` condition
requires *both* pairs to differ — the message is slightly misleading if both are identical (it says
"at least one" which is still true, but imprecise).

-     assert abs(X_filled_1[1, 0] - X_filled_2[1, 0]) > 1e-12 and abs(X_filled_1[3, 0] - X_filled_2[3, 0]) > 1e-12, (
-         "Stochastic imputations should differ at both missing positions; at least one pair was identical"
-     )
+     # Use seeded imputers with different random states for a deterministic divergence check
+     imputer_a = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=42)
+     imputer_b = NufiImputer(method='direct', alpha=1e-4, covariance_compensation=False, random_state=99)
+     imputer_a.fit(X)
+     imputer_b.fit(X)
+     X_a = imputer_a.transform(X, stochastic=True, stochastic_scale=1.5)
+     X_b = imputer_b.transform(X, stochastic=True, stochastic_scale=1.5)
+     assert not np.array_equal(X_a, X_b), "Different random seeds should produce different stochastic imputations"
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>31</id>
    <title>test_imputer.py:136-137 - GCV quality bounds too lenient — poor imputations pass undetected.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:136-137

**GCV quality bounds too lenient — poor imputations pass undetected.**

The bounds check in `test_gcv_tuning` uses `obs_min - 0.5*col_range` to `obs_max + 0.5*col_range`,
which allows imputed values up to 50% outside the observed min/max. A degenerate imputer that fills
every NaN with the column mean (or a constant near the mean) would easily pass this check, masking
regressions in the GCV auto-tuning logic. For a tuned model, imputed values should stay within or
very close to the observed range. Consider tightening the tolerance (e.g., `obs_min -
0.05*col_range` to `obs_max + 0.05*col_range`) or adding a variance/continuity check.

-         assert np.all(X_filled[:, col_idx] >= obs_min - 0.5 * col_range)
-         assert np.all(X_filled[:, col_idx] <= obs_max + 0.5 * col_range)
+         # Tighten bounds to catch obviously poor imputations
+         assert np.all(X_filled[:, col_idx] >= obs_min - 0.1 * col_range)
+         assert np.all(X_filled[:, col_idx] <= obs_max + 0.1 * col_range)
+         # Additional sanity: imputed values should not all be identical to the mean
+         nan_rows = np.isnan(col_obs)
+         if nan_rows.sum() > 1:
+             assert np.std(X_filled[nan_rows, col_idx]) > 1e-8, (
+                 f"Imputed values in column {col_idx} appear constant"
+             )
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>32</id>
    <title>test_imputer.py:212-213 - Overly broad warning match pattern risks false positives.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:212-213

**Overly broad warning match pattern risks false positives.**

The regex pattern `'all-NaN|empty|no valid'` uses very generic substrings (`empty`, `no valid`) that
could match unrelated warnings from NumPy, pandas, or other dependencies (e.g., "empty slice", "no
valid index"). If a dependency starts emitting warnings containing these words — for example, a
`FutureWarning` about an "empty" DataFrame operation — this test will silently pass even if the
expected imputer warning is never raised. Use a more specific pattern tied to the actual warning
message text, or at minimum anchor the alternatives with word boundaries.

-     with pytest.warns(UserWarning, match="all-NaN|empty|no valid"):
+     with pytest.warns(UserWarning, match=r"all.NaN|column.*empty|no valid (observations|values|samples)"):
          X_filled = imputer1.fit_transform(X_all_nan)
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>33</id>
    <title>test_imputer.py:64-77 - MultiIndex wrapper test covers only the simplest grouping scenario.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:64-77

**MultiIndex wrapper test covers only the simplest grouping scenario.**

This test uses only two groups (`group_A`, `group_B`) with identical `[1.0, 2.0]` time indices in
each group, and a single value column. Real-world multi-index data often has varying group sizes,
non-aligned timestamps across groups, and multiple signal columns. This limited coverage could hide
bugs such as incorrect group isolation, cross-group data leakage, or failures with unaligned index
levels. Add test cases with:
- Groups of different sizes (e.g., 3 rows in group_A, 5 rows in group_B)
- Different timestamp values per group
- Multiple value columns
- Groups with all-NaN or all-observed columns

  def test_multiindex_wrapper():
      arrays = [
          ['group_A', 'group_A', 'group_B', 'group_B'],
          [1.0, 2.0, 1.0, 2.0]
      ]
      index = pd.MultiIndex.from_arrays(arrays, names=('entity', 'time'))
      df = pd.DataFrame({'signal': [1.5, np.nan, np.nan, 3.5]}, index=index)
      
      imputer = NufiImputer(method='direct', covariance_compensation=False)
      df_filled = infill_multiindex_dataframe(df, imputer)
      
      assert isinstance(df_filled, pd.DataFrame)
      assert not df_filled.isna().any().any()
      assert df_filled.loc[('group_A', 1.0), 'signal'] == 1.5
+ 
+ 
+ def test_multiindex_wrapper_uneven_groups():
+     # Groups with different sizes and non-aligned timestamps
+     arrays = [
+         ['group_A', 'group_A', 'group_A', 'group_B', 'group_B'],
+         [1.0, 3.0, 5.0, 2.0, 4.0]
+     ]
+     index = pd.MultiIndex.from_arrays(arrays, names=('entity', 'time'))
+     df = pd.DataFrame({
+         'signal': [1.0, np.nan, 5.0, np.nan, 4.0]
+     }, index=index)
+     
+     imputer = NufiImputer(method='direct', covariance_compensation=False)
+     df_filled = infill_multiindex_dataframe(df, imputer)
+     assert isinstance(df_filled, pd.DataFrame)
+     assert not df_filled.isna().any().any()
+     # Values in different groups should be imputed independently
+     assert df_filled.loc[('group_A', 1.0), 'signal'] == 1.0
+     assert df_filled.loc[('group_B', 4.0), 'signal'] == 4.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>34</id>
    <title>test_imputer.py:15-23 - Infill wrapper test does not verify imputation quality.</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/tests/test_imputer.py:15-23

**Infill wrapper test does not verify imputation quality.**

`test_infill_dataframe_wrapper` only asserts that no NaN remains and that observed values are
preserved. A bug that replaces every NaN with an arbitrary constant (e.g., `0.0` or `999.0`) would
pass undetected. Consider adding a sanity check: for a simple linear-like signal, the filled values
should not be wildly inconsistent with the observed trend (e.g., `10.0, NaN, 30.0` → filled value
should be roughly near `20.0`, not `-1000.0`).

      # Infill without keeping time_col as feature
      df_filled = infill_dataframe(df, time_col='timestamp', keep_time_col=False)
      assert isinstance(df_filled, pd.DataFrame)
      assert not df_filled.isna().any().any()
      assert len(df_filled) == len(df)
      assert 'timestamp' not in df_filled.columns
      # Verify observed values are preserved
      assert df_filled.loc[0, 'signal'] == 10.0
      assert df_filled.loc[2, 'signal'] == 30.0
+     # Sanity: filled values should be within a reasonable range of observed data
+     assert 5.0 <= df_filled.loc[1, 'signal'] <= 35.0, (
+         f"Filled value {df_filled.loc[1, 'signal']} is far outside observed range [10, 30]"
+     )
+     assert 5.0 <= df_filled.loc[3, 'signal'] <= 35.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>35</id>
    <title>agent.py:122-124 - Bug: list_versions crashes on CSV files with no underscore</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:122-124

**Bug: `list_versions` crashes on CSV files with no underscore**

The `else` branch handles files where `len(parts) < 3` by indexing `parts[1]`, but if a non-tracker
`.csv` file (e.g., `data.csv`) exists in `.nufi_history/`, `parts` has only one element and
`parts[1]` raises `IndexError`, crashing `list_versions`, `revert_to_version`, and any caller.

**Fix suggestion**: Either skip files that don't match the expected naming pattern, or guard with
`if len(parts) < 2: continue`.

                  else:
-                     version_id = parts[0]
-                     step_name = parts[1].replace(".csv", "")
+                     # Unknown naming pattern; skip safely
+                     continue
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>36</id>
    <title>agent.py:600-608 - Bug: plot_diagnostics does not sort timestamps before computing Nyquist frequency and PSD</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:600-608

**Bug: `plot_diagnostics` does not sort timestamps before computing Nyquist frequency and PSD**

Unlike `impute_dataframe` (which sorts `v_timestamps` and `v_data` before computing `np.diff`),
`plot_diagnostics` skips sorting entirely. Unsorted timestamps produce negative `np.diff` values,
causing `np.nanmin(p_n) > 0` to be `False`, which silently falls back to `min_p = 1.0` — giving a
wrong Nyquist frequency of `0.5 Hz`. The unsorted timestamps are also passed directly to
`solve_tikhonov_nudft`, which may produce incorrect spectra.

**Fix suggestion**: Add sorting before the `np.diff` call, matching the logic in `impute_dataframe`:

          valid_mask = ~np.isnan(orig_data) & ~np.isnan(timestamps)
          v_timestamps = timestamps[valid_mask]
          v_data = orig_data[valid_mask]
+         
+         # Ensure sorted before computing sampling intervals
+         if len(v_timestamps) > 1 and not np.all(np.diff(v_timestamps) >= 0):
+             sort_idx = np.argsort(v_timestamps)
+             v_timestamps = v_timestamps[sort_idx]
+             v_data = v_data[sort_idx]
          
          opt_alpha = diag.get("optimized_alpha", 1e-4)
          n_f = diag.get("n_frequencies", len(timestamps))
          
-         p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else [1.0]
-         min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
+         p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else np.array([1.0])
+         pos_mask = p_n > 0
+         min_p = np.min(p_n[pos_mask]) if np.any(pos_mask) else 1.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>37</id>
    <title>agent.py:607-608 - Bug: plot_diagnostics uses np.nanmin(p_n) instead of filtering positive diffs</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:607-608

**Bug: `plot_diagnostics` uses `np.nanmin(p_n)` instead of filtering positive diffs**

`np.nanmin(p_n)` picks the smallest diff regardless of sign. If any diff is zero or negative (common
with unsorted or duplicate timestamps), `np.nanmin(p_n) > 0` is False and `min_p` silently falls
back to `1.0`, producing an incorrect Nyquist frequency. The `impute_dataframe` diagnostics block
correctly uses `pos_mask = p_n > 0` and `np.min(p_n[pos_mask])`. This inconsistency means plots can
show wrong frequency axes.

**Fix suggestion**: Use the same positive-mask approach as `impute_dataframe` (shown together with
the previous fix above).

-         p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else [1.0]
-         min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
+         p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else np.array([1.0])
+         pos_mask = p_n > 0
+         min_p = np.min(p_n[pos_mask]) if np.any(pos_mask) else 1.0
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>39</id>
    <title>agent.py:246-254 - Design: Original index type is silently and irreversibly ...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/nufi/agent.py:246-254

Design: Original index type is silently and irreversibly mutated when time_col is None. The function
uses the existing DataFrame index and converts it to float64/int64 for NUDFT computation, then
returns the infilled DataFrame with this numeric index. The original datetime, string, or
categorical index is permanently lost. Suggestion: Store a copy of the original index before numeric
conversion, then restore it on the infilled DataFrame before returning. After the transform and
epoch restoration, restore the original index (e.g., before the return line, re-index the
DataFrame). If exact restoration is impossible, issue a clear warning.

      df_copy = df.copy()
+     original_index = df_copy.index.copy()  # Preserve for restoration
      if time_col is not None:
          if time_col not in df_copy.columns:
              raise KeyError(
                  f"time_col '{time_col}' not found in DataFrame columns: {list(df_copy.columns)}"
              )
          df_copy = df_copy.set_index(time_col)
  
      if not pd.api.types.is_numeric_dtype(df_copy.index):
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>42</id>
    <title>pyproject.toml:11-11 - PEP 639 license metadata</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/pyproject.toml:11-11

**PEP 639 license metadata**: The license is defined as `{text = "MIT"}`, which is the legacy inline
format. PEP 639 (accepted) recommends using a simple string (`license = "MIT"`) along with a
`license-files` key pointing to the LICENSE file. Many automated compliance and packaging tools
(e.g., PyPI's classifier validation, SBOM generators) expect the newer format. Consider: `license =
"MIT"` and add `license-files = ["LICENSE*"]`.

- license = {text = "MIT"}
+ license = "MIT"
+ license-files = ["LICENSE*"]
]]></description>
  </task>
  <task status="WILL-NOT-DO">
    <id>44</id>
    <title>9616a63c15eb89930eeff928edf65d2033fb35:0-0 - Multiple binary Git internal object files were found (e.g.,</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/.git/objects/06/9616a63c15eb89930eeff928edf65d2033fb35:0-0

Multiple binary Git internal object files were found (e.g.,
.git/objects/06/9616a63c15eb89930eeff928edf65d2033fb35,
.git/objects/f1/87303f6a168239b76f7f65ad2b75bcf42e671b). These are compressed binary data, not
source code, and cannot be reviewed for correctness, security, performance, or maintainability.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>45</id>
    <title>.gitignore:98-98 - Maintainability: overly broad .log pattern may inadv...</title>
    <description><![CDATA[
### Location: nonuniform-fourier-infill/.gitignore:98-98

**Maintainability: overly broad `*.log` pattern may inadvertently exclude files that should be
version-controlled.**

The wildcard `*.log` ignores *all* `.log` files anywhere in the repository tree. If the project
contains test fixtures, expected-output reference files, or configuration templates with a `.log`
extension, they will be silently excluded from version control.

**Suggestion:** Consider scoping this to a specific directory (e.g., `/logs/*.log`, `logs/`) or
using anchored patterns (e.g., `/*.log`) so that only intentionally transient log files at the
repository root are ignored. Alternatively, if the project never ships `.log` files, document this
assumption clearly so future contributors are aware.
]]></description>
  </task>
</tasklist>

```