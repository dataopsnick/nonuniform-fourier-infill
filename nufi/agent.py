import os
import time
import json
import uuid
import threading
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nufi.impute import NufiImputer
from nufi.kernels.torch_kernels import get_device, solve_tikhonov_nudft

class TransformationLoggingError(Exception):
    """Custom exception raised when writing to the transformation log or history fails."""
    pass

class TransformationTracker:
    """
    Manages append-only transformation logging and snapshot-based dataframe version tracking.
    Saves snapshots under '.nufi_history/' and logs actions to 'nufi_transformations.log'.
    """

    def __init__(self, log_path: str = "nufi_transformations.log", history_dir: str = ".nufi_history"):
        self.log_path = os.path.realpath(log_path)
        self.history_dir = os.path.realpath(history_dir)
        cwd = os.path.realpath(os.getcwd())
        if not self.log_path.startswith(cwd + os.sep) and self.log_path != cwd:
            raise ValueError(f"log_path must be within current working directory: {log_path}")
        if not self.history_dir.startswith(cwd + os.sep) and self.history_dir != cwd:
            raise ValueError(f"history_dir must be within current working directory: {history_dir}")
        self._lock = threading.Lock()
        
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
        filename = f"{version_id}_{step_name}.csv"
        filepath = os.path.join(self.history_dir, filename)
        
        with self._lock:
            try:
                df.to_csv(filepath, index=True)
            except Exception as e:
                raise TransformationLoggingError(f"Failed to save data snapshot {filepath}: {e}")
            
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "snapshot_saved",
            "version_id": version_id,
            "step_name": step_name,
            "columns": list(df.columns),
            "shape": df.shape,
            "filepath": filepath
        }
        self.log_transformation(log_entry)
        return version_id

    def list_versions(self) -> list:
        """Lists all saved versions chronologically."""
        with self._lock:
            if not os.path.exists(self.history_dir):
                return []
            try:
                files = [f for f in os.listdir(self.history_dir) if f.endswith(".csv")]
                files.sort()  # Chronological order because of time-based ID prefix
                versions = []
                for f in files:
                    parts = f.split("_")
                    # Detect new format: ver_{ts}_{uuid8}_{step_name}.csv
                    if len(parts) >= 4 and len(parts[2]) == 8 and all(c in '0123456789abcdef' for c in parts[2]):
                        version_id = f"{parts[0]}_{parts[1]}_{parts[2]}"
                        step_name = "_".join(parts[3:]).replace(".csv", "")
                    elif len(parts) >= 3:
                        version_id = f"{parts[0]}_{parts[1]}"
                        step_name = "_".join(parts[2:]).replace(".csv", "")
                    else:
                        version_id = parts[0]
                        step_name = parts[1].replace(".csv", "")
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
        versions = self.list_versions()
        target = None
        for v in versions:
            if v["version_id"] == version_id:
                target = v
                break
        if target is None:
            raise ValueError(f"Version ID '{version_id}' not found in transformation history.")
            
        try:
            with self._lock:
                df = pd.read_csv(target["filepath"], index_col=0)
            
            # Log the reversion
            log_entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": "reverted_to_version",
                "version_id": version_id,
                "step_name": target["step_name"]
            }
            self.log_transformation(log_entry)
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
    tracker = TransformationTracker(log_path=log_path, history_dir=history_dir)
    
    # Take snapshot of original data
    pre_ver = tracker.save_snapshot(df, "pre_infill")

    df_copy = df.copy()
    if time_col is not None:
        df_copy = df_copy.set_index(time_col)

    if not pd.api.types.is_numeric_dtype(df_copy.index):
        raise TypeError(
            f"DataFrame index must be numeric (timestamps). "
            f"Got dtype={df_copy.index.dtype}. Provide a numeric time column via `time_col`."
        )

    timestamps = df_copy.index.to_numpy(dtype=np.float64)

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

    imputer.fit(df_copy, timestamps=timestamps)
    infilled_df = imputer.transform(df_copy, timestamps=timestamps, stochastic=stochastic, stochastic_scale=stochastic_scale)

    # Restore original index/columns name or structure if time_col was used
    if time_col is not None:
        infilled_df = infilled_df.reset_index().rename(columns={'index': time_col})

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

        p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else [1.0]
        min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
        max_sampling_rate = 1.0 / min_p
        nyquist_frequency = max_sampling_rate / 2.0
        f_k = np.linspace(0, nyquist_frequency, n_f)

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

        if imputer.covariance_compensation and imputer.d_ is not None:
            cov_scale = np.sqrt(np.abs(np.diag(imputer.d_)[col_idx]))
            if cov_scale > 0:
                reconstructed_np = reconstructed_np * cov_scale

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
    """
    orig_copy = original_df.copy()
    inf_copy = infilled_df.copy()

    if time_col is not None:
        orig_copy = orig_copy.set_index(time_col)
        inf_copy = inf_copy.set_index(time_col)

    if not pd.api.types.is_numeric_dtype(orig_copy.index):
        raise TypeError(
            f"DataFrame index must be numeric (timestamps). "
            f"Got dtype={orig_copy.index.dtype}. Provide a numeric time column via `time_col`."
        )

    timestamps = orig_copy.index.to_numpy(dtype=np.float64)

    if columns is None:
        columns = list(orig_copy.columns)[:5]

    num_cols = len(columns)
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
        
        opt_alpha = diag.get("optimized_alpha", 1e-4)
        n_f = diag.get("n_frequencies", len(timestamps))
        
        p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else [1.0]
        min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
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
