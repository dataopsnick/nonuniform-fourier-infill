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

def print_mnar_benchmark_results(results: dict):
    """Prints the MNAR dropout benchmark three-way comparison as a Markdown table."""
    print("\n" + "=" * 96)
    print("              nufi MNAR DROPOUT BENCHMARK (three-way comparison)")
    print("=" * 96)
    print(f"True treatment effect: {results['True Effect']:.5f}\n")
    header = "| {:<28} | {:<10} | {:<10} | {:<14} | {:<12} |".format(
        "Method", "Bias", "RMSE", "Cov Error", "Runtime (s)"
    )
    print(header)
    print("| {:<28} | {:<10} | {:<10} | {:<14} | {:<12} |".format(
        "-" * 28, "-" * 10, "-" * 10, "-" * 14, "-" * 12
    ))
    for name, metrics in results.items():
        if not isinstance(metrics, dict) or "Absolute Bias" not in metrics:
            continue
        print("| {:<28} | {:<10.5f} | {:<10.5f} | {:<14.5f} | {:<12.5f} |".format(
            name,
            metrics["Absolute Bias"],
            metrics["RMSE"],
            metrics["Covariance Error (Frobenius)"],
            metrics["Runtime (s)"],
        ))
    print("=" * 96 + "\n")


def _mnar_score(infilled_long, truth, missing, true_effect, final_t):
    """Scores an infilled long-format MNAR DataFrame against simulated truth.

    Returns treatment-effect bias, RMSE over the post-dropout (missing) cells,
    and Frobenius covariance error between the two arms' final-visit outcomes.
    """
    def _final_by_arm(arm):
        mask = (infilled_long["arm"] == arm) & (infilled_long["time"] == final_t)
        return infilled_long.loc[mask, "outcome"].to_numpy(dtype=float)

    est_effect = float(np.mean(_final_by_arm("Treatment")) - np.mean(_final_by_arm("Control")))
    bias = abs(est_effect - true_effect)

    total_sq, n = 0.0, 0
    for subj, traj in truth.items():
        sub = infilled_long[infilled_long["subject"] == subj].sort_values("time")
        pred = sub["outcome"].to_numpy(dtype=float)
        for i in missing[subj]:
            total_sq += (pred[i] - traj[i]) ** 2
            n += 1
    rmse = float(np.sqrt(total_sq / max(n, 1)))

    control_subs = sorted(k for k in truth if k.startswith("Control"))
    treatment_subs = sorted(k for k in truth if k.startswith("Treatment"))
    true_cov = np.cov(
        np.array([truth[k][-1] for k in control_subs]),
        np.array([truth[k][-1] for k in treatment_subs]),
    )
    final_by_subj = {
        row.subject: row.outcome
        for row in infilled_long[infilled_long["time"] == final_t].itertuples(index=False)
    }
    est_cov = np.cov(
        np.array([final_by_subj[k] for k in control_subs]),
        np.array([final_by_subj[k] for k in treatment_subs]),
    )
    cov_err = float(np.linalg.norm(true_cov - est_cov, ord="fro"))

    return {
        "Estimated Effect": est_effect,
        "Absolute Bias": bias,
        "RMSE": rmse,
        "Covariance Error (Frobenius)": cov_err,
    }


def run_mnar_dropout_benchmark(n_subjects_per_arm: int = 30, n_visits: int = 8, seed: int = 42) -> dict:
    """Three-way MNAR dropout comparison (Tasks 50/55).

    Compares naive MAR ``NufiImputer``, ``NufiImputer(mnar_strategy="pmm_slope")``,
    and ``infill_rbmi_dataframe(rbmi_method="CIR")``, each scored against the
    simulated truth with treatment-effect bias, RMSE, covariance error, and runtime.
    """
    from nufi.wrappers import infill_rbmi_dataframe

    rng = np.random.RandomState(seed)
    times = np.arange(n_visits, dtype=float)
    final_t = times[-1]
    records, truth, missing = [], {}, {}
    for arm, effect in (("Control", 0.0), ("Treatment", -0.6)):
        for s in range(n_subjects_per_arm):
            subj = f"{arm}_{s}"
            traj = 0.15 * times + effect * (times / max(1.0, n_visits - 1)) + rng.normal(0, 0.15, n_visits)
            # Reference-based MNAR setup: only the active (treatment) arm is
            # subject to informative dropout; the reference arm stays complete.
            if arm == "Treatment":
                threshold = np.percentile(traj, 65)
                drop_idx = next((i for i in range(2, n_visits) if traj[i] > threshold), n_visits)
            else:
                drop_idx = n_visits
            truth[subj] = traj
            missing[subj] = list(range(drop_idx, n_visits))
            for i in range(n_visits):
                records.append({
                    "subject": subj, "arm": arm, "time": times[i],
                    "outcome": traj[i] if i < drop_idx else np.nan,
                })
    long_df = pd.DataFrame(records)
    true_effect = float(
        np.mean([v[-1] for k, v in truth.items() if k.startswith("Treatment")])
        - np.mean([v[-1] for k, v in truth.items() if k.startswith("Control")])
    )

    def _infill_per_subject(imputer_factory):
        """Runs a per-subject NufiImputer over each (n_visits, 1) outcome series."""
        pieces = []
        for subj, traj in truth.items():
            sub = long_df[long_df["subject"] == subj].sort_values("time").reset_index(drop=True)
            y = sub["outcome"].to_numpy(dtype=float).reshape(-1, 1)
            imputer = imputer_factory()
            imputer.fit(y, timestamps=times)
            sub["outcome"] = imputer.transform(y, timestamps=times).ravel()
            pieces.append(sub)
        return pd.concat(pieces, ignore_index=True)

    methods = {
        "Naive MAR (NufiImputer)": lambda: _infill_per_subject(
            lambda: NufiImputer(random_state=seed)
        ),
        "PMM Slope (NufiImputer)": lambda: _infill_per_subject(
            lambda: NufiImputer(mnar_strategy="pmm_slope", random_state=seed)
        ),
        "RBMI CIR (infill_rbmi_dataframe)": lambda: infill_rbmi_dataframe(
            long_df, arm_col="arm", reference_arm="Control",
            subject_col="subject", time_col="time",
            rbmi_method="CIR", n_imputations=1, stochastic=False,
        ),
    }

    results = {
        "True Effect": true_effect,
        "n_subjects_per_arm": n_subjects_per_arm,
        "n_visits": n_visits,
    }
    for name, run in methods.items():
        start = time.time()
        infilled = run()
        runtime = float(time.time() - start)
        metrics = _mnar_score(infilled, truth, missing, true_effect, final_t)
        metrics["Runtime (s)"] = runtime
        results[name] = metrics

    return results


if __name__ == "__main__":
    benchmark_results = run_benchmarks(n_samples=250, n_channels=4, missing_rate=0.3)
    print_benchmark_results(benchmark_results)

    mnar_results = run_mnar_dropout_benchmark(n_subjects_per_arm=30, n_visits=8)
    print_mnar_benchmark_results(mnar_results)
    benchmark_results["MNAR Dropout"] = mnar_results

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
