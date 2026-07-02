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
from nufi.impute import NufiImputer
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

    def __init__(self, log_path: str = "nufi_transformations.log", history_dir: str = ".nufi_history"):
        safe_root = os.path.realpath(os.getcwd())
        # Resolve paths and validate immediately:
        for p in (log_path, history_dir):
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
    history_dir: str = ".nufi_history"
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

    tracker = TransformationTracker(log_path=log_path, history_dir=history_dir)
    
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
            reconstructed_np = imputer.reconstructed_[col_idx][valid_mask]
            F_np = imputer.coefficients_[col_idx]
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