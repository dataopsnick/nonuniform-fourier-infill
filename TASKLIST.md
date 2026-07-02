<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>torch_kernels.py:113-116 - Output length mismatch (breaking API change).</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:113-116

**Output length mismatch (breaking API change).** The original implementation always returned a
tensor of length `len(data)` (= `len(X.data)`, which includes NaN positions). The new code defines
`N = len(v_data)` (only valid, non‑NaN points) and returns a tensor of that length. Callers that
assume the output length matches the input signal length — for indexing, alignment, or downstream
covariance tracking — will get silently truncated results or index errors. While the function
appears experimental (not yet called in the current codebase), this must be reconciled before
integration. Either restore `N = len(data)` and zero‑pad the output, or clearly document the new
contract.

+         N_full = len(data)
          N = len(v_data)
          if N == 0:
-             results.append(torch.zeros(0, dtype=torch.complex128, device=dev))
+             results.append(torch.zeros(N_full, dtype=torch.complex128, device=dev))
              continue

**Agreement / Rationale:**
We fully agree with this recommendation. To maintain compatibility with `compute_ND_NUDFT` and prevent silent truncation or errors in downstream callers (such as `covariance_compensation` which asserts equal output length across multi-dimensional signals), the function must return a tensor matching the full input signal length, or zero-pad the invalid positions.
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>torch_kernels.py:161-162 - Missing grid oversampling normalization factor.</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:161-162

**Missing grid oversampling normalization factor.** After the FFT-based NUDFT, the result should be
scaled by `1/M` (or a related factor) to account for the oversampled grid. Without this, the output
amplitude will be systematically scaled by `~M` relative to a direct O(N²) NUDFT summation.
Additionally, the sum of the spreading weights per sample is not normalized, so different timestamp
distributions will produce different effective scales. Add `F = F_grid[:N] * apodization / M` (or
apply a constant normalization factor consistent with the chosen NUDFT convention).

-         # Extract the positive frequencies and apply the analytic correction
-         F = F_grid[:N] * apodization
+         # Extract the positive frequencies, apply analytic correction and grid normalisation
+         F = F_grid[:N] * apodization / M

**Agreement / Rationale:**
We agree with the grievance that the fast gridding-based transform output currently lacks proper scale normalization to match the direct O(N^2) NUDFT summation. Reconciling this scaling (whether via `1/M`, dividing by the sum of the weights, or other normalization factors) is required to ensure consistent amplitudes between the direct and fast NUDFT methods.
]]></description>
  </task>
  <task status="AGREED">
    <id>4</id>
    <title>torch_kernels.py:89-103 - nyquist_frequency parameter is accepted but never used in the algorithm.</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:89-103

**`nyquist_frequency` parameter is accepted but never used in the algorithm.** The warning on line
100–103 suggests the parameter affects behavior, but the gridding computation (coordinate scaling,
grid size, spreading, FFT length) depends solely on the number of valid time-domain points `N`.
Users passing an explicit `nyquist_frequency` will get the same output as if they had not. Either
remove the parameter to avoid misleading callers, or use it to control the output frequency range
(e.g., adjust the grid size `M` or the number of extracted frequency bins).

      def compute_Fast_ND_NUDFT(X_list, device=None, nyquist_frequency=None):
      """
      Replication of FINUFFT's 'Gaussian Spreading' method
      
      Type-1 Fast Non-Uniform DFT using Gaussian Gridding.
      Achieves O(N log N) complexity while perfectly preserving C^infinity continuity 
      and preventing the spectral leakage caused by linear interpolation.
      
+     .. note::
+         The ``nyquist_frequency`` parameter is reserved for future use and is
+         currently ignored by the Gaussian gridding algorithm.
      """
      dev = get_device(device)
      results = []
      
-     if nyquist_frequency is None and len(X_list) > 1:
+     _ = nyquist_frequency  # reserved; not yet wired into the gridding algorithm
+     if len(X_list) > 1:
          import warnings
-         warnings.warn("nyquist_frequency not provided; estimating per-signal. "
-                       "Pass an explicit nyquist_frequency for multi-signal workflows.")
+         warnings.warn("nyquist_frequency is not yet used by the Fast NUDFT algorithm. "
+                       "Output frequency scaling depends solely on the number of valid points.")

**Agreement / Rationale:**
We agree that accepting a parameter that is completely unused and has no effect on the underlying gridding computation is misleading and should be clearly documented, warned, or integrated.
]]></description>
  </task>
  <task status="AGREED">
    <id>5</id>
    <title>torch_kernels.py:121-121 - Degenerate input when all valid timestamps are identical.</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:121-121

**Degenerate input when all valid timestamps are identical.** When `t_min == t_max`, `span` is
silently set to `1.0`, mapping every non-uniform point to `t_scaled = 0`. All samples are spread
onto the same grid position, producing a flat (constant) spectrum. This may not be a useful result,
and callers may not realise the data is degenerate. Consider emitting a warning or raising an error
when `span` is forced to `1.0`.

+         if t_max <= t_min:
+             import warnings
+             warnings.warn("All non-NaN timestamps are identical; spectrum will be degenerate.")
          span = t_max - t_min if t_max > t_min else 1.0

**Agreement / Rationale:**
We agree with this recommendation. Emitting a warning when all non-NaN timestamps are identical helps callers identify degenerate inputs that produce a constant/flat spectrum.
]]></description>
  </task>
</tasklist>