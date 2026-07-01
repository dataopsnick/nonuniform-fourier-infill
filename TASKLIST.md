```xml
<tasklist>
  <task status="COMPLETED">
    <id>1</id>
    <title>setup.py:79-79 - The sys.prefix candidate checks os.path.isdir(os.path....</title>
    <description><![CDATA[
### Location: setup.py:79-79

The `sys.prefix` candidate checks `os.path.isdir(os.path.join(candidate, "include"))` (line 89).
Since `sys.prefix/include` exists in virtually every Python environment (it contains Python
headers), this candidate will always match — even when libomp is not installed there. Because
`sys.prefix` appears before `/usr/local` and `/usr` in the candidate list, the loop will break early
and never reach those fallbacks. Consider either checking for a more specific marker (e.g.,
`omp.h`), or adding a library existence check (e.g., `os.path.exists(os.path.join(candidate, "lib",
"libomp.dylib"))`) alongside the include check.

              sys.prefix,  # conda / virtualenv (include/ and lib/ live directly under prefix)
+             os.path.join(sys.prefix, "lib"),  # some virtualenv layouts
]]></description>
  </task>
  <task status="COMPLETED">
    <id>2</id>
    <title>impute.py:156-158 - perm_small has length len(valid_idx_comp), but actua...</title>
    <description><![CDATA[
### Location: nufi/impute.py:156-158

`perm_small` has length `len(valid_idx_comp)`, but `actual_valid_cols` has length
`len(signal_valid_idx)` which may be smaller due to `np.unique` collapsing real+imag pairs of the
same column. This can cause an `IndexError` at lines 161, 166, and 170 when both real (2*j) and imag
(2*j+1) components of the same column survive in `valid_idx_comp`. Consider defining
`actual_valid_cols` with the same length as `valid_idx_comp`:
```
actual_valid_cols = [valid_cols[idx // 2] for idx in valid_idx_comp]
```
This preserves the 1:1 correspondence that `perm_small` expects.

-                 # Filter valid_cols using valid_idx_comp to handle degenerate columns dropped
-                 signal_valid_idx = np.unique(valid_idx_comp // 2)
-                 actual_valid_cols = [valid_cols[idx] for idx in signal_valid_idx]
+                 # Filter valid_cols using valid_idx_comp to handle degenerate columns dropped.
+                 # valid_idx_comp references the doubled (real+imag) space; map back via // 2.
+                 actual_valid_cols = [valid_cols[idx // 2] for idx in valid_idx_comp]
]]></description>
  </task>
</tasklist>

```