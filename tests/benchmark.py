import os
import time
import json
import numpy as np
import pandas as pd
from nufi.impute import NufiImputer

# Enable IterativeImputer in scikit-learn (MICE)
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    MICE_AVAILABLE = True
except ImportError:
    MICE_AVAILABLE = False

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

def generate_benchmark_data(n_samples: int = 200, n_channels: int = 3, missing_rate: float = 0.3, seed: int = 42):
    """
    Generates non-uniformly sampled synthetic multi-channel time-series data
    with known underlying correlation and smooth derivative structure.
    """
    np.random.seed(seed)
    
    # 1. Non-uniform timestamps
    timestamps = np.sort(np.random.uniform(0, 10.0, n_samples))
    
    # 2. Correlated ground truth signals using a shared latent factor
    latent = 2.0 * np.sin(1.5 * timestamps) + np.cos(3.0 * timestamps)
    
    ground_truth = np.zeros((n_samples, n_channels))
    for c in range(n_channels):
        scale = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, np.pi)
        noise = np.random.normal(0, 0.1, n_samples)
        ground_truth[:, c] = scale * np.sin(timestamps + phase) + 0.5 * latent + noise
        
    # 3. Introduce random missingness (NaNs)
    masked_data = ground_truth.copy()
    mask = np.random.rand(n_samples, n_channels) < missing_rate
    masked_data[mask] = np.nan
    
    # NOTE: Boundary filling intentionally skipped to avoid biasing benchmarks.
    # If a method requires non-NaN endpoints (e.g., cubic spline), handle it
    # inside that method's try-block only.
    pass
            
    df_truth = pd.DataFrame(ground_truth, index=timestamps, columns=[f"ch_{i}" for i in range(n_channels)])
    df_masked = pd.DataFrame(masked_data, index=timestamps, columns=[f"ch_{i}" for i in range(n_channels)])
    
    return timestamps, df_truth, df_masked

def run_benchmarks(n_samples: int = 200, n_channels: int = 3, missing_rate: float = 0.3, gp_max_valid: int = 500) -> dict:
    """Runs empirical comparison of NUFI, MICE, GPs, and Splines."""
    timestamps, df_truth, df_masked = generate_benchmark_data(n_samples, n_channels, missing_rate)
    
    results = {}
    true_cov = df_truth.cov().to_numpy()
    
    # ==========================================
    # 1. NUFI Imputer (Our Library)
    # ==========================================
    start = time.time()
    try:
        nufi = NufiImputer(device='cpu', covariance_compensation=True, n_frequencies='auto', alpha='auto', random_state=42)
        # timestamps are already the DataFrame index; do not pass duplicate kwarg
        nufi_infilled = nufi.fit_transform(df_masked)
        nufi_time = time.time() - start
        
        if np.any(np.isnan(nufi_infilled.to_numpy())):
            raise ValueError("Imputation returned NaN values.")
        nufi_rmse = np.sqrt(np.mean((df_truth.to_numpy() - nufi_infilled.to_numpy()) ** 2))
        try:
            cov_diff = true_cov - nufi_infilled.cov().to_numpy()
            if np.any(np.isnan(cov_diff)):
                nufi_cov_err = float('nan')
            else:
                nufi_cov_err = np.linalg.norm(cov_diff, ord='fro')
        except (ValueError, np.linalg.LinAlgError):
            nufi_cov_err = float('nan')
        
        results["NUFI"] = {
            "RMSE": float(nufi_rmse),
            "Covariance Error (Frobenius)": float(nufi_cov_err),
            "Runtime (s)": float(nufi_time)
        }
    except Exception as e:
        import traceback
        print(f"[WARN] NUFI benchmark failed: {e}")
        traceback.print_exc()
        results["NUFI"] = {"Error": f"{type(e).__name__}: {e}"}

    # ==========================================
    # 2. Cubic Spline Interpolation
    # ==========================================
    start = time.time()
    try:
        spline_infilled = df_masked.interpolate(method='cubic', axis=0)
        remaining_nan = spline_infilled.isna().sum().sum()
        if remaining_nan > 0:
            print(f"[WARN] Cubic spline left {remaining_nan} NaN(s) — results may be degraded by fallback")
            # Fallback: use linear, then ffill/bfill to ensure no NaN left
            spline_infilled = spline_infilled.interpolate(method='linear', axis=0).ffill().bfill()
        spline_time = time.time() - start
        
        if np.any(np.isnan(spline_infilled.to_numpy())):
            raise ValueError("Imputation returned NaN values.")
        spline_rmse = np.sqrt(np.mean((df_truth.to_numpy() - spline_infilled.to_numpy()) ** 2))
        try:
            cov_diff = true_cov - spline_infilled.cov().to_numpy()
            if np.any(np.isnan(cov_diff)):
                spline_cov_err = float('nan')
            else:
                spline_cov_err = np.linalg.norm(cov_diff, ord='fro')
        except (ValueError, np.linalg.LinAlgError):
            spline_cov_err = float('nan')
        
        results["Cubic Spline"] = {
            "RMSE": float(spline_rmse),
            "Covariance Error (Frobenius)": float(spline_cov_err),
            "Runtime (s)": float(spline_time)
        }
    except Exception as e:
        results["Cubic Spline"] = {"Error": str(e)}

    # ==========================================
    # 3. MICE (IterativeImputer)
    # ==========================================
    if MICE_AVAILABLE:
        start = time.time()
        try:
            # Normalize timestamps to [0,1] so MICE regularization treats them comparably
            t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
            combined_masked = np.hstack([t_norm.reshape(-1, 1), df_masked.to_numpy()])
            
            mice = IterativeImputer(max_iter=10, random_state=42)
            mice_infilled_combined = mice.fit_transform(combined_masked)
            mice_infilled_data = mice_infilled_combined[:, 1:]
            mice_time = time.time() - start
            
            if np.any(np.isnan(mice_infilled_data)):
                raise ValueError("Imputation returned NaN values.")
            mice_rmse = np.sqrt(np.mean((df_truth.to_numpy() - mice_infilled_data) ** 2))
            mice_infilled_df = pd.DataFrame(mice_infilled_data, columns=df_truth.columns)
            try:
                cov_diff = true_cov - mice_infilled_df.cov().to_numpy()
                if np.any(np.isnan(cov_diff)):
                    mice_cov_err = float('nan')
                else:
                    mice_cov_err = np.linalg.norm(cov_diff, ord='fro')
            except (ValueError, np.linalg.LinAlgError):
                mice_cov_err = float('nan')
            
            results["MICE"] = {
                "RMSE": float(mice_rmse),
                "Covariance Error (Frobenius)": float(mice_cov_err),
                "Runtime (s)": float(mice_time)
            }
        except Exception as e:
            results["MICE"] = {"Error": str(e)}
    else:
        results["MICE"] = {"Status": "Not Available (Scikit-Learn experimental features missing)"}

    # ==========================================
    # 4. Gaussian Process (GP) Regression
    # ==========================================
    if GP_AVAILABLE:
        start = time.time()
        try:
            gp_infilled_data = np.zeros_like(df_truth.to_numpy())
            any_skipped = False
            for c in range(n_channels):
                col_data = df_masked.to_numpy()[:, c]
                valid = ~np.isnan(col_data)
                n_valid = valid.sum()
                if n_valid > gp_max_valid:
                    print(f"[INFO] Skipping GP for channel {c} due to too many valid points ({n_valid} > {gp_max_valid})")
                    any_skipped = True
                    col_series = pd.Series(col_data)
                    gp_infilled_data[:, c] = col_series.interpolate(method='linear').ffill().bfill().to_numpy()
                    continue
                if n_valid < 2:
                    # Not enough observations to fit a GP; fill with NaN to avoid biased scores
                    gp_infilled_data[:, c] = np.nan if n_valid == 0 else np.nanmean(col_data)
                    continue
                
                gp = GaussianProcessRegressor(
                    kernel=RBF(length_scale=np.ptp(timestamps) / np.sqrt(len(timestamps))),
                    alpha=0.1,
                    random_state=42,
                    n_restarts_optimizer=3
                )
                gp.fit(timestamps[valid].reshape(-1, 1), col_data[valid])
                gp_infilled_data[:, c] = gp.predict(timestamps.reshape(-1, 1))
                
            gp_time = time.time() - start
            if np.any(np.isnan(gp_infilled_data)):
                raise ValueError("Imputation returned NaN values.")
            gp_rmse = np.sqrt(np.mean((df_truth.to_numpy() - gp_infilled_data) ** 2))
            gp_infilled_df = pd.DataFrame(gp_infilled_data, columns=df_truth.columns)
            try:
                cov_diff = true_cov - gp_infilled_df.cov().to_numpy()
                if np.any(np.isnan(cov_diff)):
                    gp_cov_err = float('nan')
                else:
                    gp_cov_err = np.linalg.norm(cov_diff, ord='fro')
            except (ValueError, np.linalg.LinAlgError):
                gp_cov_err = float('nan')
            
            results["Gaussian Process"] = {
                "RMSE": float(gp_rmse),
                "Covariance Error (Frobenius)": float(gp_cov_err),
                "Runtime (s)": float(gp_time)
            }
            if any_skipped:
                results["Gaussian Process"]["Status"] = "Completed with some channels skipped (linear fallback)"
        except Exception as e:
            results["Gaussian Process"] = {"Error": str(e)}
    else:
        results["Gaussian Process"] = {"Status": "Not Available (Scikit-Learn missing)"}

    return results

def print_benchmark_results(results: dict):
    """Prints the benchmark results in a beautiful Markdown Table."""
    print("\n" + "=" * 80)
    print("                      nufi EMPIRICAL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"| {'Method':<20} | {'RMSE':<10} | {'Covariance Error':<20} | {'Runtime (s)':<12} |")
    print(f"| {'-'*20} | {'-'*10} | {'-'*20} | {'-'*12} |")
    
    for method, metrics in results.items():
        if "RMSE" in metrics:
            print(f"| {method:<20} | {metrics['RMSE']:<10.5f} | {metrics['Covariance Error (Frobenius)']:<20.5f} | {metrics['Runtime (s)']:<12.5f} |")
        else:
            status = metrics.get("Status", metrics.get("Error", "Error"))
            print(f"| {method:<20} | {status:<49} |")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    benchmark_results = run_benchmarks(n_samples=250, n_channels=4, missing_rate=0.3)
    print_benchmark_results(benchmark_results)

    # Save results to disk
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{ts}.json"
        with open(filename, "w") as f:
            json.dump(benchmark_results, f, indent=4)
            print(f"Results saved to {filename}")
    except IOError as e:
        print(f"[WARN] Could not save results to disk: {e}")
