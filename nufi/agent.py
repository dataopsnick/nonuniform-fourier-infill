import os
import time
import json
import uuid
import threading
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nufi.impute import (
    NufiImputer,
    pool_rubins_rules,
    compute_mnar_offsets,
    evaluate_fourier_derivative,
)
from nufi.kernels.torch_kernels import get_device, solve_tikhonov_nudft

class TransformationLoggingError(Exception):
    """Custom exception raised when writing to the transformation log or history fails."""
    pass

_FILE_LOCKS = {}
_FILE_LOCKS_LOCK = threading.Lock()

class TransformationTracker:
    """
    Manages append-only transformation logging and snapshot-based dataframe version tracking.
    Saves snapshots under '.nufi_history/' and logs actions to 'nufi_transformations.log'.
    """

    def __init__(self, log_path: str = "nufi_transformations.log", history_dir: str = ".nufi_history",
                 allow_outside_cwd: bool = None):
        if allow_outside_cwd is None:
            allow_outside_cwd = os.environ.get(
                "NUFI_ALLOW_UNSAFE_PATHS", ""
            ).strip().lower() in ("1", "true", "yes")
        safe_root = os.path.realpath(os.getcwd())
        # Resolve paths and validate immediately:
        for p in (log_path, history_dir):
            if allow_outside_cwd:
                break
            resolved = os.path.realpath(p)
            try:
                if os.path.commonpath([safe_root, resolved]) != safe_root:
                    raise ValueError(f"Path {p} is outside the allowed directory.")
            except ValueError:
                raise ValueError(f"Path {p} is outside the allowed directory (possibly on a different drive).")
        # Store resolved paths (these are canonical and safe for writes)
        self.log_path = os.path.realpath(log_path)
        self.history_dir = os.path.realpath(history_dir)
        
        with _FILE_LOCKS_LOCK:
            key = (self.log_path, self.history_dir)
            if key not in _FILE_LOCKS:
                _FILE_LOCKS[key] = threading.RLock()
            self._lock = _FILE_LOCKS[key]
        # NOTE: This lock is thread-safe only. Concurrent processes sharing the same
        # log/history paths will corrupt files. Use file locking or dedicated IPC
        # if cross-process safety is required.
        
        with self._lock:
            try:
                os.makedirs(self.history_dir, exist_ok=True)
            except Exception as e:
                raise TransformationLoggingError(f"Failed to create history directory: {e}")

    def log_transformation(self, log_entry: dict):
        """Appends a transformation log entry to the log file in append-only mode."""
        with self._lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                raise TransformationLoggingError(f"Failed to write to transformation log: {e}")

    def save_snapshot(self, df: pd.DataFrame, step_name: str) -> str:
        """Saves a dataframe snapshot with timestamp and unique version ID."""
        version_id = f"ver_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        # Use parquet for I/O performance, dtype preservation, and storage efficiency
        filename = f"{version_id}_{step_name}.parquet"
        filepath = os.path.join(self.history_dir, filename)
        
        with self._lock:
            # Write parquet first (if it fails, no log pollution)
            try:
                df.to_parquet(filepath, engine='pyarrow', compression='snappy', index=True)
            except Exception as e:
                raise TransformationLoggingError(f"Failed to save data snapshot {filepath}: {e}")
            # Log only after successful write to avoid orphan entries
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "snapshot_saved",
                "version_id": version_id,
                "step_name": step_name,
                "columns": list(df.columns),
                "shape": df.shape,
                "filepath": filepath
            }
            try:
                self.log_transformation(log_entry)
            except Exception as e:
                # Clean up orphaned parquet to keep history consistent
                try:
                    os.remove(filepath)
                except OSError as rm_err:
                    import warnings
                    warnings.warn(
                        f"Failed to remove orphaned snapshot {filepath} after log write failure: {rm_err}. "
                        f"Manual cleanup may be required.",
                        UserWarning
                    )
                raise TransformationLoggingError(f"Failed to write to transformation log: {e}")
        return version_id

    def list_versions(self) -> list:
        """Lists all saved versions chronologically."""
        with self._lock:
            if not os.path.exists(self.history_dir):
                return []
            try:
                files = [f for f in os.listdir(self.history_dir) if f.endswith(".parquet")]
                files.sort()  # Chronological order because of time-based ID prefix
                versions = []
                for f in files:
                    parts = f.split("_")
                    if len(parts) >= 4 and re.match(r'^[0-9a-f]{8}$', parts[2]):
                        version_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                        if not re.match(r'^ver_\d+_[0-9a-f]{8}$', version_id):
                            continue
                        step_name = "_".join(parts[3:]).replace(".parquet", "")
                    elif len(parts) >= 3:
                        version_id = f"{parts[0]}_{parts[1]}"
                        if not re.match(r'^ver_\d+$', version_id):
                            continue
                        step_name = "_".join(parts[2:]).replace(".parquet", "")
                    else:
                        # Unknown naming pattern; skip safely
                        continue
                    versions.append({
                        "version_id": version_id,
                        "step_name": step_name,
                        "filename": f,
                        "filepath": os.path.join(self.history_dir, f)
                    })
                return versions
            except Exception as e:
                raise TransformationLoggingError(f"Failed to list history directory: {e}")

    def revert_to_version(self, version_id: str) -> pd.DataFrame:
        """Loads and returns a saved dataframe snapshot of the specified version_id."""
        with self._lock:
            versions = self.list_versions()
            target = None
            for v in versions:
                if v["version_id"] == version_id:
                    target = v
                    break
            if target is None:
                raise ValueError(f"Version ID '{version_id}' not found in transformation history.")
                
            try:
                if not os.path.exists(target["filepath"]):
                    raise TransformationLoggingError(f"Snapshot file {target['filepath']} no longer exists.")
                df = pd.read_parquet(target["filepath"])
                # Validate snapshot integrity
                if df.empty:
                    raise TransformationLoggingError(f"Snapshot {version_id} is empty or corrupted.")
            
                # Log the reversion
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "event": "reverted_to_version",
                    "version_id": version_id,
                    "step_name": target["step_name"]
                }
                try:
                    self.log_transformation(log_entry)
                except TransformationLoggingError:
                    import warnings
                    warnings.warn(
                        f"Failed to log reversion to {version_id}; audit trail may be incomplete.",
                        UserWarning
                    )
                return df
            except Exception as e:
                raise TransformationLoggingError(f"Failed to load or log reverted version {version_id}: {e}")

def impute_dataframe(
    df: pd.DataFrame,
    time_col: str = None,
    method: str = 'direct',
    device: str = None,
    n_frequencies: str = 'auto',
    alpha: str = 'auto',
    solver: str = 'direct',
    covariance_compensation: bool = True,
    stochastic: bool = False,
    stochastic_scale: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    log_path: str = "nufi_transformations.log",
    history_dir: str = ".nufi_history",
    allow_outside_cwd: bool = None
) -> tuple[pd.DataFrame, dict]:
    """
    High-level, zero-config entrypoint for infilling a Pandas DataFrame.
    Automatically configures optimal hyperparameters, performs infilling,
    maintains persistent append-only logs, and tracks data lineage for DVC/Git-style reversion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with missing values (NaNs).
    time_col : str, optional
        Name of the column containing timestamps. If None, the DataFrame index is used.
    method : str, default='direct'
        Fourier computation method.
    device : str, optional
        Hardware accelerator device.
    n_frequencies : str or int, default='auto'
        Number of frequencies. If 'auto', Optimized via GCV.
    alpha : str or float, default='auto'
        Regularization penalty. If 'auto', optimized via GCV.
    solver : str, default='direct'
        Linear system solver ('direct' or 'cg').
    covariance_compensation : bool, default=True
        Whether to perform LDL^T covariance compensation.
    stochastic : bool, default=False
        Whether to perform stochastic multiple imputation (Bayesian infilling).
    stochastic_scale : float, default=1.0
        Scaling factor for stochastic noise.
    max_iter : int, default=100
        Maximum iterations for the CG solver.
    tol : float, default=1e-5
        Tolerance parameter for CG convergence.
    log_path : str, default="nufi_transformations.log"
        Path to the transaction log file.
    history_dir : str, default=".nufi_history"
        Directory to save dataframe snapshots.
    allow_outside_cwd : bool, optional
        When True, skip the containment guard that normally requires `log_path`
        and `history_dir` to resolve inside the current working directory. When
        None (default), the guard is controlled by the ``NUFI_ALLOW_UNSAFE_PATHS``
        environment variable.

    Returns
    -------
    infilled_df : pd.DataFrame
        Infilled Pandas DataFrame.
    diagnostics : dict
        Rich diagnostic metadata dictionary.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df cannot be empty")
    if time_col is not None and list(df.columns).count(time_col) > 1:
        raise ValueError(f"time_col '{time_col}' appears multiple times in DataFrame columns. "
                         f"Ensure column names are unique.")

    tracker = TransformationTracker(log_path=log_path, history_dir=history_dir,
                                    allow_outside_cwd=allow_outside_cwd)
    
    # Take snapshot of original data
    pre_ver = tracker.save_snapshot(df, "pre_infill")

    df_copy = df.copy()
    original_index = df_copy.index.copy()  # Preserve for restoration
    if time_col is not None:
        if time_col not in df_copy.columns:
            raise KeyError(
                f"time_col '{time_col}' not found in DataFrame columns: {list(df_copy.columns)}"
            )
        df_copy = df_copy.set_index(time_col)

    if not pd.api.types.is_numeric_dtype(df_copy.index):
        try:
            # Handle datetime-like index explicitly
            if pd.api.types.is_datetime64_any_dtype(df_copy.index):
                import warnings
                warnings.warn(
                    "DatetimeIndex detected. Converting to nanosecond epoch (int64) which may cause precision loss.",
                    UserWarning
                )
                df_copy.index = df_copy.index.astype(np.int64)
            else:
                numeric_idx = pd.to_numeric(df_copy.index, errors='coerce')
                if numeric_idx.isna().any():
                    raise ValueError("Index contains non-convertible values")
                if np.can_cast(numeric_idx, np.int64, casting='safe'):
                    df_copy.index = numeric_idx.astype(np.int64)
                else:
                    # float64 can lose precision for integers > 2^53.
                    # Consider normalising timestamps (e.g., t - t[0]) for large values.
                    if np.issubdtype(numeric_idx.dtype, np.integer) and numeric_idx.max() > 2**53:
                        import warnings
                        warnings.warn(
                            "Timestamp values exceed 2^53; float64 conversion may lose precision. "
                            "Consider normalising timestamps or using a smaller unit.",
                            UserWarning
                        )
                    df_copy.index = numeric_idx.astype(np.float64)
        except Exception:
            raise TypeError(
                f"DataFrame index must be numeric (timestamps). "
                f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col` "
                f"or ensure your index contains numeric values."
            )

    timestamps = df_copy.index.to_numpy(dtype=np.float64)
    max_ts = np.max(np.abs(timestamps)) if len(timestamps) > 0 else 0
    epoch = 0.0
    if max_ts > 2**53:
        import warnings
        # Subtract epoch to preserve relative precision in float64
        epoch = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        df_copy.index = df_copy.index - epoch
        timestamps = df_copy.index.to_numpy(dtype=np.float64)
        warnings.warn(
            f"Timestamps exceed float64 precision (max={max_ts:.1e}). "
            f"Normalized by subtracting epoch={epoch} to preserve relative precision."
        )

    # Initialize and fit NufiImputer
    imputer = NufiImputer(
        method=method,
        device=device,
        covariance_compensation=covariance_compensation,
        n_frequencies=n_frequencies,
        alpha=alpha,
        solver=solver,
        max_iter=max_iter,
        tol=tol
    )

    try:
        imputer.fit(df_copy, timestamps=timestamps)
        infilled_df = imputer.transform(df_copy, timestamps=timestamps, stochastic=stochastic, stochastic_scale=stochastic_scale)
    except Exception:
        try:
            tracker.log_transformation({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "infilled_dataframe_failed",
                "pre_infill_version": pre_ver,
                "parameters": {
                    "method": method,
                    "device": str(device),
                    "n_frequencies": n_frequencies,
                    "alpha": alpha,
                    "solver": solver,
                    "covariance_compensation": covariance_compensation,
                    "stochastic": stochastic,
                    "stochastic_scale": stochastic_scale
                }
            })
        except Exception:
            pass  # do not mask the original exception
        raise

    # Restore the epoch after transform to map back to original unshifted index
    if epoch != 0.0:
        infilled_df.index = infilled_df.index + epoch

    # Restore original index/columns name or structure if time_col was used
    if time_col is not None:
        infilled_df = infilled_df.reset_index().rename(columns={'index': time_col})
    else:
        infilled_df.index = original_index

    # Generate JSON diagnostic metadata and column details
    diagnostics = {}
    dev = get_device(device)
    columns = [c for c in df.columns if c != time_col]
    total_infilled_nans = int(df_copy[columns].isna().sum().sum())

    for col_idx, col_name in enumerate(columns):
        col_data = df_copy[col_name].to_numpy()
        valid_mask = ~np.isnan(col_data) & ~np.isnan(timestamps)
        v_timestamps = timestamps[valid_mask]
        v_data = col_data[valid_mask]
        N_val = len(v_data)

        if N_val == 0:
            diagnostics[col_name] = {
                "snr_db": None,
                "spectral_entropy": None,
                "normalized_spectral_entropy": None,
                "stability_flags": ["NO_OBSERVATIONS"],
                "optimized_alpha": float(imputer.alphas_[col_idx]) if col_idx < len(imputer.alphas_) else None,
                "n_frequencies": int(imputer.n_frequencies_[col_idx]) if col_idx < len(imputer.n_frequencies_) else None
            }
            continue

        opt_alpha = imputer.alphas_[col_idx] if col_idx < len(imputer.alphas_) else 1e-4
        n_f = imputer.n_frequencies_[col_idx] if col_idx < len(imputer.n_frequencies_) else len(col_data)

        if len(v_timestamps) > 1:
            # Ensure sorted before computing sampling intervals
            if not np.all(np.diff(v_timestamps) >= 0):
                sort_idx = np.argsort(v_timestamps)
                v_timestamps = v_timestamps[sort_idx]
                v_data = v_data[sort_idx]
        p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else np.array([1.0])
        pos_mask = p_n > 0
        min_p = np.min(p_n[pos_mask]) if np.any(pos_mask) else 1.0
        max_sampling_rate = 1.0 / min_p
        nyquist_frequency = max_sampling_rate / 2.0
        f_k = np.linspace(0, nyquist_frequency, n_f)

        if len(v_data) > 0 and hasattr(imputer, 'reconstructed_') and col_idx in imputer.reconstructed_:
            recon_valid = np.asarray(imputer.reconstructed_[col_idx])[valid_mask]
            valid_ts = np.asarray(timestamps)[valid_mask]
            reconstructed_np = recon_valid[np.argsort(valid_ts)]
            F_np = np.asarray(imputer.coefficients_[col_idx])
        else:
            # WARNING: fallback PSD does NOT apply covariance compensation.
            # SNR/entropy/flags may differ from the actual compensated imputation.
            F = solve_tikhonov_nudft(
                v_timestamps, v_data, f_k, opt_alpha,
                solver=solver, max_iter=max_iter, tol=tol, device=device
            )
            F_np = F.cpu().numpy()

            t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
            t_times = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
            exponent = 2.0j * np.pi * t_times.unsqueeze(1) * t_f_k.unsqueeze(0)
            reconstructed = torch.real(torch.sum(F.unsqueeze(0) * torch.exp(exponent), dim=1))
            reconstructed_np = reconstructed.cpu().numpy()

        # Note: imputer.reconstructed_ already includes covariance compensation.
        # Do not apply cov_scale again to avoid double-compensation in diagnostics.

        signal_variance = np.var(reconstructed_np)
        residual = v_data - reconstructed_np
        residual_variance = np.var(residual)

        if residual_variance < 1e-12:
            snr_db = 100.0
        else:
            snr_db = float(10 * np.log10(max(1e-12, signal_variance) / max(1e-12, residual_variance)))

        psd = np.abs(F_np) ** 2
        psd_sum = np.sum(psd)
        if psd_sum > 0:
            p = psd / psd_sum
            entropy = -np.sum(p * np.log(p + 1e-15))
            normalized_entropy = entropy / np.log(len(F_np)) if len(F_np) > 1 else 0.0
        else:
            entropy = 0.0
            normalized_entropy = 0.0

        flags = []
        missing_percentage = (np.isnan(col_data).sum() / len(col_data)) * 100.0
        if missing_percentage > 80.0:
            flags.append("HIGH_MISSINGNESS")
        if N_val < 10:
            flags.append("LOW_OBSERVATIONS")
        if opt_alpha <= 1e-6:
            flags.append("POTENTIAL_OVERFIT_LOW_REGULARIZATION")
        elif opt_alpha >= 5.0:
            flags.append("HIGH_REGULARIZATION")

        # --- Diggle-Kenward pre-dropout MNAR diagnostics (Task 50) ---
        nan_mask_col = np.isnan(col_data)
        pre_dropout_idx = np.where(~nan_mask_col[:-1] & nan_mask_col[1:])[0]
        if len(pre_dropout_idx) >= 2 and N_val >= 6:
            dx_dt = evaluate_fourier_derivative(F_np, f_k, timestamps)
            obs_mask_col = ~nan_mask_col & ~np.isnan(timestamps)
            obs_level = col_data[obs_mask_col]
            level_std = float(np.std(obs_level))
            if (
                obs_level.size > 0
                and level_std > 0.0
                and abs(np.mean(col_data[pre_dropout_idx]) - np.mean(obs_level))
                > 1.5 * level_std
            ):
                flags.append("MNAR_DROPOUT_LEVEL_BIAS")
            obs_deriv = dx_dt[obs_mask_col]
            if obs_deriv.size >= 2:
                deriv_std = float(np.std(obs_deriv))
                if (
                    deriv_std > 0.0
                    and abs(np.mean(dx_dt[pre_dropout_idx]) - np.mean(obs_deriv))
                    > 1.5 * deriv_std
                ):
                    flags.append("MNAR_DROPOUT_SLOPE_BIAS")

        if not flags:
            flags.append("STABLE")

        diagnostics[col_name] = {
            "snr_db": snr_db,
            "spectral_entropy": float(entropy),
            "normalized_spectral_entropy": float(normalized_entropy),
            "stability_flags": flags,
            "optimized_alpha": float(opt_alpha),
            "n_frequencies": int(n_f)
        }

    # Take snapshot of infilled data
    post_ver = tracker.save_snapshot(infilled_df, "post_infill")

    # Save the main transaction log
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": "infilled_dataframe_success",
        "pre_infill_version": pre_ver,
        "post_infill_version": post_ver,
        "infilled_nans": total_infilled_nans,
        "parameters": {
            "method": method,
            "device": str(device),
            "n_frequencies": n_frequencies,
            "alpha": alpha,
            "solver": solver,
            "covariance_compensation": covariance_compensation,
            "stochastic": stochastic,
            "stochastic_scale": stochastic_scale
        },
        "diagnostics": diagnostics
    }
    tracker.log_transformation(log_entry)

    return infilled_df, diagnostics

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

    Parameters
    ----------
    original_df : pd.DataFrame
        Original DataFrame containing observed data (with NaNs).
    infilled_df : pd.DataFrame
        Infilled DataFrame containing fully imputed data.
    diagnostics : dict
        Rich diagnostic metadata dictionary returned by `impute_dataframe`.
    time_col : str, optional
        Name of the column containing timestamps. If None, the DataFrame index is used.
    columns : list, optional
        List of column names to visualize. If None, defaults to the first 5 columns.
    save_path : str, optional
        Path to save the generated visualization. If None, the plot is not saved.
    show_plot : bool, default=True
        Whether to display the generated plot.
    solver : str, default='direct'
        Linear system solver for PSD computation ('direct' or 'cg').
    max_iter : int, default=100
        Maximum iterations for CG solver.
    tol : float, default=1e-5
        Tolerance for CG solver convergence.
    device : str, optional
        Hardware accelerator device for PSD computation.
    """
    orig_copy = original_df.copy()
    inf_copy = infilled_df.copy()

    if time_col is not None:
        orig_copy = orig_copy.set_index(time_col)
        inf_copy = inf_copy.set_index(time_col)

    for df_copy in (orig_copy, inf_copy):
        if not pd.api.types.is_numeric_dtype(df_copy.index):
            try:
                # Handle datetime-like index explicitly
                if pd.api.types.is_datetime64_any_dtype(df_copy.index):
                    import warnings
                    warnings.warn(
                        "DatetimeIndex detected. Converting to nanosecond epoch (int64) which may cause precision loss.",
                        UserWarning
                    )
                    df_copy.index = df_copy.index.astype(np.int64)
                else:
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

    timestamps = orig_copy.index.to_numpy(dtype=np.float64)
    max_ts = np.max(np.abs(timestamps)) if len(timestamps) > 0 else 0
    epoch = 0.0
    if max_ts > 2**53:
        import warnings
        epoch = float(timestamps[0]) if len(timestamps) > 0 else 0.0
        orig_copy.index = orig_copy.index - epoch
        inf_copy.index = inf_copy.index - epoch
        timestamps = orig_copy.index.to_numpy(dtype=np.float64)
        warnings.warn(
            f"Timestamps exceed float64 precision (max={max_ts:.1e}). "
            f"Normalized by subtracting epoch={epoch} to preserve relative precision."
        )

    if columns is None:
        columns = list(orig_copy.columns)[:5]
    else:
        missing = [c for c in columns if c not in orig_copy.columns]
        if missing:
            import warnings
            warnings.warn(f"Requested columns not found in DataFrame: {missing}")
            columns = [c for c in columns if c in orig_copy.columns]

    num_cols = len(columns)
    if num_cols == 0:
        import warnings
        warnings.warn("No columns to plot. Returning empty figure.")
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        axes = np.array([[axes]])
        return fig, axes
    fig, axes = plt.subplots(num_cols, 2, figsize=(14, 4 * num_cols), squeeze=False)

    for idx, col_name in enumerate(columns):
        orig_data = orig_copy[col_name].to_numpy()
        inf_data = inf_copy[col_name].to_numpy()
        diag = diagnostics.get(col_name, {})

        # Plot 1: Time Domain Reconstruction
        ax_time = axes[idx, 0]
        ax_time.plot(timestamps, inf_data, label='Infilled Smooth Signal', color='#1f77b4', linewidth=2)
        ax_time.scatter(timestamps, orig_data, label='Observed Data', color='#ff7f0e', s=25, zorder=5)
        ax_time.set_title(f"Time Domain: {col_name}")
        ax_time.set_xlabel("Time")
        ax_time.set_ylabel("Amplitude")
        ax_time.grid(True, linestyle='--', alpha=0.6)
        ax_time.legend()

        # Plot 2: Frequency Domain Spectrum & Diagnostics Text
        ax_freq = axes[idx, 1]
        
        valid_mask = ~np.isnan(orig_data) & ~np.isnan(timestamps)
        v_timestamps = timestamps[valid_mask]
        v_data = orig_data[valid_mask]
        
        # Ensure sorted before computing sampling intervals
        if len(v_timestamps) > 1 and not np.all(np.diff(v_timestamps) >= 0):
            sort_idx = np.argsort(v_timestamps)
            v_timestamps = v_timestamps[sort_idx]
            v_data = v_data[sort_idx]
            
        opt_alpha = diag.get("optimized_alpha", 1e-4)
        n_f = diag.get("n_frequencies", len(timestamps))
        
        p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else np.array([1.0])
        pos_mask = p_n > 0
        min_p = np.min(p_n[pos_mask]) if np.any(pos_mask) else 1.0
        max_sampling_rate = 1.0 / min_p
        nyquist_frequency = max_sampling_rate / 2.0
        f_k = np.linspace(0, nyquist_frequency, n_f)
        
        if len(v_data) > 0:
            F = solve_tikhonov_nudft(
                v_timestamps, v_data, f_k, opt_alpha,
                solver=solver, max_iter=max_iter, tol=tol, device=device
            )
            F_np = F.cpu().numpy()
            psd = np.abs(F_np) ** 2
            ax_freq.stem(f_k, psd, linefmt='g-', markerfmt='go', basefmt='r-', label='Power Spectrum')
        else:
            ax_freq.text(0.5, 0.5, "No observations to compute PSD", ha='center', va='center')

        ax_freq.set_title(f"PSD & Diagnostics: {col_name}")
        ax_freq.set_xlabel("Frequency (Hz)")
        ax_freq.set_ylabel("Power")
        ax_freq.grid(True, linestyle='--', alpha=0.6)

        # Draw diagnostic box
        snr = diag.get("snr_db", 0.0)
        ent = diag.get("normalized_spectral_entropy", 0.0)
        flags = ", ".join(diag.get("stability_flags", ["UNKNOWN"]))
        
        textstr = '\n'.join((
            f"SNR: {snr:.2f} dB" if snr is not None else "SNR: N/A",
            f"Spectral Entropy: {ent:.3f}" if ent is not None else "Entropy: N/A",
            f"Flags: {flags}",
            f"Alpha: {opt_alpha:.2e}"
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax_freq.text(0.95, 0.95, textstr, transform=ax_freq.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    else:
        plt.close()
    return fig, axes


def _parse_delta_grid(delta_grid):
    """Normalizes delta_grid into (control_deltas, treatment_deltas) 1-D axes."""
    if (
        isinstance(delta_grid, (tuple, list))
        and len(delta_grid) == 2
        and all(isinstance(g, (list, np.ndarray)) for g in delta_grid)
    ):
        control = np.unique(np.asarray(delta_grid[0], dtype=np.float64).ravel())
        treatment = np.unique(np.asarray(delta_grid[1], dtype=np.float64).ravel())
        return control, treatment
    arr = np.asarray(delta_grid, dtype=np.float64)
    if arr.ndim == 1:
        return np.array([0.0]), np.unique(arr)
    if arr.ndim == 2:
        return np.unique(arr[:, 0]), np.unique(arr[0, :])
    raise ValueError(
        "delta_grid must be a 1-D array, a 2-D (n_control, n_treatment) array, "
        "or a tuple of two 1-D arrays."
    )


def run_tipping_point_analysis(
    df: pd.DataFrame,
    arm_col: str,
    outcome_col: str,
    delta_grid,
    time_col: str = None,
    n_imputations: int = 50,
    alpha_level: float = 0.05,
    subject_col: str = None,
    reference_arm=None,
    mnar_strategy: str = "pmm_shift",
    random_state=None,
    log_path: str = "nufi_transformations.log",
    history_dir: str = ".nufi_history",
    allow_outside_cwd: bool = None,
) -> dict:
    """
    Executes a vectorized MNAR Tipping Point Sensitivity Analysis over delta_grid
    without redundant NUDFT spectral solves, pooling p-values via Rubin's Rules
    to identify the tipping point boundary delta* (Task 48).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if df.empty:
        raise ValueError("df cannot be empty")
    for col in (arm_col, outcome_col):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame columns: {list(df.columns)}")
    if subject_col is not None and subject_col not in df.columns:
        raise KeyError(f"subject_col '{subject_col}' not found in DataFrame columns: {list(df.columns)}")
    if time_col is not None and time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not found in DataFrame columns: {list(df.columns)}")
    if not isinstance(n_imputations, (int, np.integer)) or n_imputations < 2:
        raise ValueError("n_imputations must be an integer >= 2 for Rubin's Rules pooling.")
    if not (0.0 < float(alpha_level) < 1.0):
        raise ValueError(f"alpha_level must be in (0, 1); got {alpha_level}")
    if mnar_strategy not in ("pmm_shift", "pmm_slope"):
        raise ValueError("run_tipping_point_analysis supports 'pmm_shift' or 'pmm_slope'.")

    control_deltas, treatment_deltas = _parse_delta_grid(delta_grid)
    if control_deltas.size == 0 or treatment_deltas.size == 0:
        raise ValueError("delta_grid must contain at least one value per axis.")
    if not (np.all(np.isfinite(control_deltas)) and np.all(np.isfinite(treatment_deltas))):
        raise ValueError("delta_grid must contain only finite values.")

    arm_values = pd.unique(df[arm_col])
    if reference_arm is not None:
        if reference_arm not in arm_values:
            raise ValueError(f"reference_arm {reference_arm!r} not found in '{arm_col}'.")
        control_arm = reference_arm
        others = [a for a in arm_values if a != reference_arm]
        if not others:
            raise ValueError("No treatment arm available besides the reference arm.")
        treatment_arm = others[0]
    else:
        if len(arm_values) < 2:
            raise ValueError(f"Tipping point analysis requires at least two arms in '{arm_col}'.")
        control_arm, treatment_arm = arm_values[0], arm_values[1]

    from sklearn.utils import check_random_state
    rng = check_random_state(random_state)

    tracker = TransformationTracker(log_path=log_path, history_dir=history_dir,
                                    allow_outside_cwd=allow_outside_cwd)
    pre_ver = tracker.save_snapshot(df, "pre_tipping_point")

    def _resolve_times(arm_df):
        if time_col is not None:
            t_vals = pd.to_numeric(arm_df[time_col], errors="coerce")
            if t_vals.isna().any():
                raise ValueError(f"Column '{time_col}' contains non-numeric / NaN timestamps.")
            return t_vals.to_numpy(dtype=np.float64)
        try:
            t_vals = arm_df.index.to_numpy(dtype=np.float64)
        except (TypeError, ValueError):
            raise TypeError("DataFrame index must be numeric when time_col is None.")
        if np.isnan(t_vals).any():
            raise ValueError("Index contains NaN timestamps.")
        return t_vals

    def _fit_arm(arm_label):
        arm_df = df[df[arm_col] == arm_label]
        if arm_df.empty:
            raise ValueError(f"Arm {arm_label!r} has no rows.")
        t_vals = _resolve_times(arm_df)
        subj_vals = (
            arm_df[subject_col].to_numpy()
            if subject_col is not None
            else np.full(len(arm_df), "__arm_series__")
        )
        wide = pd.DataFrame(
            {"time": t_vals, "subject": subj_vals, "value": arm_df[outcome_col].to_numpy(dtype=np.float64)}
        )
        pivot = wide.pivot_table(index="time", columns="subject", values="value", aggfunc="mean").sort_index()
        pivot = pivot.dropna(axis=1, how="all")
        if pivot.empty:
            raise ValueError(f"Arm {arm_label!r} has no observed outcome values.")
        times_arm = pivot.index.to_numpy(dtype=np.float64)
        matrix = pivot.to_numpy(dtype=np.float64)

        arm_imputer = NufiImputer(
            n_frequencies="auto",
            alpha="auto",
            random_state=int(rng.randint(0, 2**31 - 1)),
        )
        arm_imputer.fit(pivot, timestamps=times_arm)
        base = arm_imputer.transform(pivot, timestamps=times_arm, stochastic=False).to_numpy(dtype=np.float64)
        noise = arm_imputer.sample_posterior_noise(len(times_arm), int(n_imputations), rng=rng)
        if noise.ndim == 2:
            noise = noise[None, :, :]

        miss_mask = np.isnan(matrix)
        offsets_unit = np.zeros_like(matrix)
        for j in range(matrix.shape[1]):
            offsets_unit[:, j] = compute_mnar_offsets(mnar_strategy, 1.0, times_arm, miss_mask[:, j])

        return {
            "base_mean": base.mean(axis=0),
            "unit_mean": offsets_unit.mean(axis=0),
            "noise_mean": (noise * miss_mask.astype(np.float64)[None, :, :]).mean(axis=1),
            "n_subjects": matrix.shape[1],
        }

    stats_control = _fit_arm(control_arm)
    stats_treatment = _fit_arm(treatment_arm)

    C_mesh, T_mesh = np.meshgrid(control_deltas, treatment_deltas, indexing="ij")
    effects = np.zeros(C_mesh.shape)
    p_values = np.zeros(C_mesh.shape)

    base_c = stats_control["base_mean"][None, :]
    unit_c = stats_control["unit_mean"][None, :]
    noise_c = stats_control["noise_mean"]
    base_t = stats_treatment["base_mean"][None, :]
    unit_t = stats_treatment["unit_mean"][None, :]
    noise_t = stats_treatment["noise_mean"]
    n_c = stats_control["n_subjects"]
    n_t = stats_treatment["n_subjects"]

    for i, d_c in enumerate(control_deltas):
        endpoints_c = base_c + d_c * unit_c + noise_c
        var_c = endpoints_c.var(axis=1, ddof=1) / n_c if n_c > 1 else np.full(n_imputations, 1e-6)
        mean_c = endpoints_c.mean(axis=1)
        for j, d_t in enumerate(treatment_deltas):
            endpoints_t = base_t + d_t * unit_t + noise_t
            var_t = endpoints_t.var(axis=1, ddof=1) / n_t if n_t > 1 else np.full(n_imputations, 1e-6)
            q = endpoints_t.mean(axis=1) - mean_c
            u = var_t + var_c
            pooled = pool_rubins_rules(q, u, n_obs=max(n_c + n_t, 3))
            effects[i, j] = pooled["pooled_estimate"]
            p_values[i, j] = pooled["p_value"]

    significant = p_values < alpha_level
    tipping_point = None
    if (~significant).any():
        dist = np.sqrt(C_mesh**2 + T_mesh**2)
        dist_masked = np.where(~significant, dist, np.inf)
        i0, j0 = np.unravel_index(np.argmin(dist_masked), dist.shape)
        tipping_point = {
            "control_delta": float(C_mesh[i0, j0]),
            "treatment_delta": float(T_mesh[i0, j0]),
            "pooled_effect": float(effects[i0, j0]),
            "p_value": float(p_values[i0, j0]),
        }

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": "tipping_point_analysis",
        "pre_infill_version": pre_ver,
        "parameters": {
            "arm_col": arm_col,
            "outcome_col": outcome_col,
            "control_arm": str(control_arm),
            "treatment_arm": str(treatment_arm),
            "mnar_strategy": mnar_strategy,
            "n_imputations": int(n_imputations),
            "alpha_level": float(alpha_level),
            "grid_shape": [int(effects.shape[0]), int(effects.shape[1])],
            "control_deltas": [float(x) for x in control_deltas],
            "treatment_deltas": [float(x) for x in treatment_deltas],
        },
        "tipping_point": tipping_point,
    }
    tracker.log_transformation(log_entry)

    return {
        "control_deltas": control_deltas,
        "treatment_deltas": treatment_deltas,
        "treatment_effects": effects,
        "p_values": p_values,
        "significant": significant,
        "tipping_point": tipping_point,
        "alpha_level": float(alpha_level),
        "n_imputations": int(n_imputations),
        "mnar_strategy": mnar_strategy,
        "arms": {"control": control_arm, "treatment": treatment_arm},
    }


def plot_tipping_point(results, save_path=None, show_plot=True, cmap="RdBu_r", ax=None):
    """
    Renders a publication-ready contour/heatmap of pooled treatment effects and
    p-value significance boundaries from run_tipping_point_analysis (Task 48).
    """
    required = ("control_deltas", "treatment_deltas", "treatment_effects", "p_values", "alpha_level")
    missing = [k for k in required if k not in results]
    if missing:
        raise ValueError(f"results dict is missing required keys: {missing}")

    c = np.asarray(results["control_deltas"], dtype=np.float64).ravel()
    t = np.asarray(results["treatment_deltas"], dtype=np.float64).ravel()
    effects = np.asarray(results["treatment_effects"], dtype=np.float64)
    p_values = np.asarray(results["p_values"], dtype=np.float64)
    alpha = float(results["alpha_level"])
    tipping = results.get("tipping_point")

    if effects.size == 0 or (len(c) == 1 and len(t) == 1):
        raise ValueError("delta_grid must contain at least two coordinates to plot.")

    if len(c) == 1 or len(t) == 1:
        if len(c) == 1:
            x, eff, pv = t, effects.reshape(len(c), len(t))[0], p_values.reshape(len(c), len(t))[0]
            xlabel = "Treatment-arm delta"
        else:
            x, eff, pv = c, effects.reshape(len(c), len(t))[:, 0], p_values.reshape(len(c), len(t))[:, 0]
            xlabel = "Control-arm delta"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, eff, color="#1f77b4", lw=2, marker="o", label="Pooled treatment effect")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Pooled treatment effect", color="#1f77b4")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax2 = ax.twinx()
        ax2.plot(x, pv, color="#d62728", lw=2, marker="s", label="Pooled p-value")
        ax2.axhline(alpha, color="#d62728", ls="--", lw=1.2)
        ax2.set_ylabel("Pooled p-value", color="#d62728")
        if tipping is not None:
            tip_x = tipping["treatment_delta"] if len(c) == 1 else tipping["control_delta"]
            ax.axvline(tip_x, color="black", ls=":", lw=1.5)
        ax.set_title("Tipping Point Sensitivity Analysis (1-D sweep)")
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        cf = ax.contourf(t, c, effects, levels=20, cmap=cmap)
        fig.colorbar(cf, ax=ax, label="Pooled treatment effect")
        try:
            cs = ax.contour(t, c, p_values, levels=[alpha], colors="black", linewidths=1.6)
            ax.clabel(cs, fmt={alpha: f"p = {alpha:g}"})
        except Exception:
            pass
        if tipping is not None:
            ax.scatter(
                [tipping["treatment_delta"]],
                [tipping["control_delta"]],
                marker="X",
                s=140,
                color="black",
                zorder=5,
                label=f"delta* = ({tipping['control_delta']:g}, {tipping['treatment_delta']:g})",
            )
            ax.legend(loc="upper right")
        ax.set_xlabel("Treatment-arm delta")
        ax.set_ylabel("Control-arm delta")
        ax.set_title("Tipping Point Sensitivity Analysis")

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
