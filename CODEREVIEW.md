[ocr] 1 file(s) changed, reviewing 1 in /home/user/app/nonuniform-fourier-infill
[ocr] Plan completed for nufi/kernels/torch_kernels.py
[ocr]   ▶ code_search search_text=compute_Fast_ND_NUDFT
[ocr]   ✔ code_search (4ms)
[ocr]   ▶ file_find query_name=test_torch_kernels
[ocr]   ✔ file_find (5ms)
[ocr]   ▶ code_search search_text=nyquist_frequency
[ocr]   ✔ code_search (2ms)
[ocr]   ▶ file_read file_path=nufi/kernels/torch_kernels.py
[ocr]   ✔ file_read (9ms)
[ocr]   ▶ file_find query_name=torch_kernels
[ocr]   ✔ file_find (4ms)
[ocr]   ▶ code_search search_text=compute_Fast_ND_NUDFT file_patterns=[*.py]
[ocr]   ✔ code_search (2ms)
[ocr]   ▶ file_find query_name=test
[ocr]   ✔ file_find (2ms)
[ocr]   ▶ code_search search_text=Fast_ND_NUDFT file_patterns=[*.py]
[ocr]   ✔ code_search (2ms)
[ocr]   ▶ code_search search_text=scatter_add_ file_patterns=[*.py]
[ocr]   ✔ code_search (1ms)
[ocr]   ▶ code_search search_text=Fast_ND_NUDFT
[ocr]   ✔ code_search (2ms)
[ocr]   ▶ file_read file_path=tests/test_covariance.py
[ocr]   ✔ file_read (3ms)
[ocr]   ▶ file_read file_path=tests/test_agent.py
[ocr]   ✔ file_read (3ms)
[ocr]   ▶ file_find query_name=impute
[ocr]   ✔ file_find (4ms)
[ocr]   ▶ file_find query_name=agent
[ocr]   ✔ file_find (4ms)
[ocr]   ▶ file_read file_path=nufi/impute.py
[ocr]   ✔ file_read (4ms)
[ocr]   ▶ file_read file_path=tests/test_imputer.py
[ocr]   ✔ file_read (3ms)
[ocr]   ▶ file_read file_path=nufi/agent.py start_line=1 end_line=100
[ocr]   ✔ file_read (2ms)
[ocr]   ▶ file_find query_name=wrappers
[ocr]   ✔ file_find (2ms)
[ocr]   ▶ file_read end_line=300 file_path=nufi/agent.py start_line=100
[ocr]   ✔ file_read (2ms)
[ocr]   ▶ file_read start_line=300 end_line=500 file_path=nufi/agent.py
[ocr]   ✔ file_read (2ms)
[ocr]   ▶ file_read file_path=nufi/wrappers.py
[ocr]   ✔ file_read (2ms)
[ocr]   ▶ file_find query_name=requirements
[ocr]   ✔ file_find (3ms)
[ocr]   ▶ file_find query_name=pyproject
[ocr]   ✔ file_find (2ms)
[ocr]   ▶ file_read file_path=requirements.txt
[ocr]   ✔ file_read (2ms)
[ocr]   ▶ file_read file_path=pyproject.toml
[ocr]   ✔ file_read (2ms)
[ocr]   ▶ file_read file_path=nufi/kernels/torch_kernels.py start_line=89 end_line=170
[ocr]   ✔ file_read (6ms)
[ocr]   ▶ code_comment "nufi/kernels/torch_kernels.py"
[ocr]   ✔ code_comment (0s)
[ocr]   ▶ code_comment "nufi/kernels/torch_kernels.py"
[ocr]   ✔ code_comment (1ms)
[ocr]   ▶ code_comment "nufi/kernels/torch_kernels.py"
[ocr]   ✔ code_comment (0s)
[ocr]   ▶ code_comment "nufi/kernels/torch_kernels.py"
[ocr]   ✔ code_comment (0s)
[ocr]   ▶ code_comment "nufi/kernels/torch_kernels.py"
[ocr]   ✔ code_comment (1ms)
[ocr] Summary: 1 file(s) reviewed, 5 comment(s), ~525585 token(s) used (input: ~511933, output: ~13652), 4m17s elapsed

─── nufi/kernels/torch_kernels.py:113-116 ───
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


─── nufi/kernels/torch_kernels.py:137-140 ───
**Asymmetric Gaussian spreading window causes systematic spectral bias.** `grid_idx` is computed
with `torch.floor`, so `grid_pos` always lies in `[grid_idx, grid_idx+1)`. The window `range(-W//2 +
1, W//2 + 1)` = `[-2, -1, 0, 1, 2, 3]` (6 points) is not centered on the true continuous position.
For example, with `grid_pos = 7.3`, `grid_idx = 7`, the window visits `[5,6,7,8,9,10]` — it extends
3 points above the floored index but only 2 below, when it should be symmetric. This shifts the
effective centroid of the spread and introduces a frequency-dependent phase error.

Fix: use `torch.round` to find the nearest grid point, then use a symmetric window `range(-W//2,
W//2)` = `[-3, -2, -1, 0, 1, 2]`.

-         grid_idx = torch.floor(grid_pos).to(torch.long)
+         grid_idx = torch.round(grid_pos).to(torch.long)
  
          # Smear the amplitude across the +/- 3 neighboring grid points
-         for w in range(-W//2 + 1, W//2 + 1):
+         for w in range(-W//2, W//2):


─── nufi/kernels/torch_kernels.py:161-162 ───
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


─── nufi/kernels/torch_kernels.py:89-103 ───
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
+     
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


─── nufi/kernels/torch_kernels.py:121-121 ───
**Degenerate input when all valid timestamps are identical.** When `t_min == t_max`, `span` is
silently set to `1.0`, mapping every non-uniform point to `t_scaled = 0`. All samples are spread
onto the same grid position, producing a flat (constant) spectrum. This may not be a useful result,
and callers may not realise the data is degenerate. Consider emitting a warning or raising an error
when `span` is forced to `1.0`.

+         if t_max <= t_min:
+             import warnings
+             warnings.warn("All non-NaN timestamps are identical; spectrum will be degenerate.")
          span = t_max - t_min if t_max > t_min else 1.0



Processing review output into task list...

✅ **Successfully generated XML task list with 5 items.**
Copy the block below for issue import:
