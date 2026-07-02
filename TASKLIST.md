```xml
<tasklist>
  <task status="NOT STARTED">
    <id>1</id>
    <title>agent.py:68-75 - The filename is constructed with a .csv extension and t...</title>
    <description><![CDATA[
### Location: nufi/agent.py:68-75

The filename is constructed with a `.csv` extension and then immediately replaced with `.parquet` on
the next line. This roundabout pattern is confusing and fragile. Consider constructing it directly
with `.parquet`.

-         filename = f"{version_id}_{step_name}.csv"
+         filename = f"{version_id}_{step_name}.parquet"
          filepath = os.path.join(self.history_dir, filename)
          
          with self._lock:
              # Write parquet first (if it fails, no log pollution)
              try:
-                 # Swap to parquet for I/O performance, dtype preservation, and storage efficiency
-                 filepath = filepath.replace('.csv', '.parquet')
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>2</id>
    <title>agent.py:92-92 - The comment still references "CSV" (the old format). This...</title>
    <description><![CDATA[
### Location: nufi/agent.py:92-92

The comment still references "CSV" (the old format). This stale reference could mislead future
readers. Consider changing to "orphaned parquet file" or simply "orphaned snapshot".

-                 # Clean up orphaned CSV to keep history consistent
+                 # Clean up orphaned parquet file to keep history consistent
]]></description>
  </task>
  <task status="NOT STARTED">
    <id>3</id>
    <title>torch_kernels.py:167-173 - Shape mismatch when N_full &gt; M (i.e., more than half the data is NaN).</title>
    <description><![CDATA[
### Location: nufi/kernels/torch_kernels.py:167-173

**Shape mismatch when N_full > M (i.e., more than half the data is NaN).**

`M = max(int(2 * N), 2)` is based on the number of *valid* samples `N`. But `k =
torch.arange(N_full, ...)` produces `N_full` elements while `F_grid[:N_full]` returns at most `M`
elements. When `N_full > M`, this causes a runtime broadcast error: `torch.exp(...)` has `N_full`
elements but `F_grid[:N_full]` has only `M` elements.

Example: N=5 valid points, N_full=20 total points → M=10, k has 20 elements, F_grid[:20] returns 10
elements → shape mismatch at line 173.

-         k = torch.arange(N_full, dtype=torch.float64, device=dev)
-         # Use a sigma scaling that matches the computed sigma from M
+         # Guard against N_full exceeding the FFT grid size M
+         n_out = min(N_full, M)
+         k = torch.arange(n_out, dtype=torch.float64, device=dev)
          apodization = torch.exp((k**2) * (sigma**2) / 2.0)
          
-         # Extract the positive frequencies, apply analytic correction and grid normalisation
-         # To maintain length matching len(data) = N_full, we slice and compute up to N_full
-         F = F_grid[:N_full] * apodization / M
+         F = F_grid[:n_out] * apodization / M
+         
+         # Pad to N_full length if needed (higher frequencies are beyond resolution)
+         if n_out < N_full:
+             pad = torch.zeros(N_full - n_out, dtype=torch.complex128, device=dev)
+             F = torch.cat([F, pad])
]]></description>
  </task>
</tasklist>

```