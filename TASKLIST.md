<tasklist>
  <task status="COMPLETED">
    <id>1</id>
    <title>agent.py:28-30 - On Windows, os.path.commonpath() raises a ValueError ...</title>
    <description><![CDATA[
### Location: nufi/agent.py:28-30

On Windows, `os.path.commonpath()` raises a `ValueError` when paths reside on different drives
(e.g., `C:\` vs `D:\`). This exception is not caught here, so users will see a cryptic `ValueError`
from `commonpath` instead of a clear, descriptive error message about the path being outside the
allowed directory.

          for path, resolved in ((log_path, resolved_log), (history_dir, resolved_hist)):
+             try:
              if os.path.commonpath([safe_root, resolved]) != safe_root:
                  raise ValueError(f"Path {path} is outside the allowed directory.")
+             except ValueError:
+                 raise ValueError(f"Path {path} is outside the allowed directory (possibly on a different drive).")
]]></description>
  </task>
  <task status="COMPLETED">
    <id>2</id>
    <title>agent.py:212-215 - The float64 cast can silently lose precision for large ...</title>
    <description><![CDATA[
### Location: nufi/agent.py:212-215

The `float64` cast can silently lose precision for large integer timestamps (e.g., nanosecond Unix
epochs). If the index contains large `int64` values exceeding 2^53, the lower bits are truncated.
Consider using `np.int64` when all converted values are integral, or warn users about precision
loss.

              numeric_idx = pd.to_numeric(df_copy.index, errors='coerce')
              if numeric_idx.isna().any():
                  raise ValueError("Index contains non-convertible values")
+             if np.can_cast(numeric_idx, np.int64, casting='safe'):
+                 df_copy.index = numeric_idx.astype(np.int64)
+             else:
              df_copy.index = numeric_idx.astype(np.float64)
]]></description>
  </task>
  <task status="COMPLETED">
    <id>3</id>
    <title>agent.py:411-425 - Only orig_copy.index is converted to numeric, but inf_...</title>
    <description><![CDATA[
### Location: nufi/agent.py:411-425

Only `orig_copy.index` is converted to numeric, but `inf_copy.index` is never checked. If
`inf_copy.index` is non-numeric (e.g., when called independently of `impute_dataframe`), the
`timestamps` used for plotting will not correctly align with `inf_data`. Apply the same conversion
to `inf_copy.index` for consistency.

-     if not pd.api.types.is_numeric_dtype(orig_copy.index):
+     for df_copy in (orig_copy, inf_copy):
+         if not pd.api.types.is_numeric_dtype(df_copy.index):
           try:
               # Attempt conversion for datetime-like or string timestamps
-             numeric_idx = pd.to_numeric(orig_copy.index, errors='coerce')
+                 numeric_idx = pd.to_numeric(df_copy.index, errors='coerce')
               if numeric_idx.isna().any():
                   raise ValueError("Index contains non-convertible values")
-             orig_copy.index = numeric_idx.astype(np.float64)
+                 df_copy.index = numeric_idx.astype(np.float64)
           except Exception:
               raise TypeError(
                   f"DataFrame index must be numeric (timestamps). "
-                 f"Got dtype={orig_copy.index.dtype}. Provide a numeric time column via `time_col` "
+                     f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col` "
                   f"or ensure your index contains numeric values."
               )
  
       timestamps = orig_copy.index.to_numpy(dtype=np.float64)
]]></description>
  </task>
</tasklist>
