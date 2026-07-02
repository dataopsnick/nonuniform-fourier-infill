import numpy as np
import torch
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from nufi.kernels.torch_kernels import get_device, compute_ND_NUDFT, compute_Fast_ND_NUDFT, covariance_compensation

class DatasetObj:
    def __init__(self, timestamps, data):
        self.timestamps = timestamps
        self.data = data

class NufiImputer(BaseEstimator, TransformerMixin):
    """
    NufiImputer uses Non-Uniform Discrete Fourier Transforms (NUDFT) to infill
    missing values (NaNs) in multidimensional time-series.
    It preserves signal covariance and derivative continuity.
    Supports L2/Tikhonov regularization, Conjugate Gradient (CG) iterative solver,
    stochastic multiple imputation, and GCV auto-tuning.
    """
    def __init__(self, method='direct', device=None, covariance_compensation=True, 
                 n_frequencies=None, alpha=1e-4, solver='direct', max_iter=100, tol=1e-5, random_state=None):
        self.method = method
        self.device = device
        self.covariance_compensation = covariance_compensation
        self.n_frequencies = n_frequencies
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.lu_ = None
        self.d_ = None
        self.perm_ = None
        self.alphas_ = []
        self.n_frequencies_ = []
        
    def clone(self):
        """Returns a fresh unfitted clone of this imputer with the same parameters."""
        return NufiImputer(
            method=self.method,
            device=self.device,
            covariance_compensation=self.covariance_compensation,
            n_frequencies=self.n_frequencies,
            alpha=self.alpha,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state
        )

    def _sort_and_compute_nyquist(self, timestamps, data):
        v_timestamps = timestamps.copy()
        v_data = data.copy()
        if len(v_timestamps) > 1:
            # Ensure sorted before computing sampling intervals
            if not np.all(np.diff(v_timestamps) >= 0):
                sort_idx = np.argsort(v_timestamps)
                v_timestamps = v_timestamps[sort_idx]
                v_data = v_data[sort_idx]
        p_n = np.diff(v_timestamps) if len(v_timestamps) > 1 else np.array([1.0])
        min_p = np.min(p_n[p_n > 0]) if np.any(p_n > 0) else 1.0
        max_sampling_rate = 1.0 / min_p
        nyquist_frequency = max_sampling_rate / 2.0
        return v_timestamps, v_data, nyquist_frequency

    def fit(self, X, y=None, timestamps=None):
        """
        Fits the imputer on X. If X is a Pandas DataFrame, the index is used as timestamps
        unless explicitly passed. Handles GCV auto-tuning for alpha and n_frequencies.
        """
        from nufi.kernels.torch_kernels import optimize_alpha_gcv, compute_gcv_from_svd
        
        # Convert X to numpy
        if isinstance(X, pd.DataFrame):
            if timestamps is None:
                timestamps = X.index.to_numpy()
            X_data = X.to_numpy()
        else:
            X_data = np.array(X)
            if timestamps is None:
                timestamps = np.arange(len(X_data), dtype=np.float64)
                
        self.timestamps_ = np.array(timestamps, dtype=np.float64)
        dev = get_device(self.device)
        
        self.alphas_ = []
        self.n_frequencies_ = []
        
        for col_idx in range(X_data.shape[1]):
            col_data = X_data[:, col_idx]
            valid_mask = ~np.isnan(col_data) & ~np.isnan(self.timestamps_)
            v_timestamps = self.timestamps_[valid_mask]
            v_data = col_data[valid_mask]
            N_val = len(v_data)
            
            if N_val == 0:
                self.alphas_.append(self.alpha if isinstance(self.alpha, (int, float)) else 1e-4)
                self.n_frequencies_.append(self.n_frequencies if isinstance(self.n_frequencies, int) else len(X_data))
                continue
            
            # Decide n_frequencies for this column
            if self.n_frequencies == 'auto':
                candidates = [max(5, N_val // 4), max(5, N_val // 2), max(5, N_val)]
                candidates = [c for c in candidates if c <= N_val]  # avoid underdetermined systems
                if not candidates:
                    candidates = [max(1, N_val)]
                candidates = sorted(list(set(candidates)))
            else:
                candidates = [self.n_frequencies if self.n_frequencies is not None else N_val]
                
            best_n_freq = candidates[0]
            best_alpha = self.alpha if isinstance(self.alpha, (int, float)) else 1e-4
            best_gcv = float('inf')
            
            v_timestamps, v_data, nyquist_frequency = self._sort_and_compute_nyquist(v_timestamps, v_data)
            
            for n_f in candidates:
                f_k = np.linspace(0, nyquist_frequency, n_f)
                t_timestamps = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
                t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
                t_data = torch.tensor(v_data, dtype=torch.float64, device=dev)
                
                exponent = 2.0j * np.pi * t_timestamps.unsqueeze(1) * t_f_k.unsqueeze(0)
                A = torch.exp(exponent)
                
                try:
                    if self.alpha == 'auto':
                        candidate_alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
                        opt_alpha, U, S, y_tilde, y_norm_sq = optimize_alpha_gcv(
                            A, t_data, candidate_alphas, return_svd=True
                        )
                    else:
                        opt_alpha = self.alpha if self.alpha is not None else 1e-4
                        # Hardware guardrail: Subsample to avoid OOM during SVD
                        MAX_SVD_SAMPLES = 2000
                        if A.shape[0] > MAX_SVD_SAMPLES:
                            sub_idx_np = rng.choice(A.shape[0], MAX_SVD_SAMPLES, replace=False)
                            sub_idx = torch.tensor(np.sort(sub_idx_np), device=dev)
                            A_sub = A[sub_idx, :]
                            y_sub = t_data[sub_idx].to(torch.complex128)
                        else:
                            A_sub, y_sub = A, t_data.to(torch.complex128)
                         
                        U, S, Vh = torch.linalg.svd(A_sub, full_matrices=False)
                        y_complex = y_sub
                        y_tilde = torch.matmul(U.adjoint(), y_complex)
                        y_norm_sq = torch.sum(torch.abs(y_complex) ** 2)
                except (RuntimeError, torch.linalg.LinAlgError, ValueError) as e:
                    import warnings
                    warnings.warn(f"SVD failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
                    continue
                
                score = compute_gcv_from_svd(S, y_tilde, y_norm_sq, opt_alpha, N_val)
                
                if score < best_gcv:
                    best_gcv = score
                    best_n_freq = n_f
                    best_alpha = opt_alpha
            
            if best_gcv == float('inf'):
                import warnings
                best_n_freq = max(1, min(5, N_val - 1)) # ensure n_f < N_val to avoid underdetermined system
                best_alpha = 1.0
                warnings.warn(
                    f"All GCV candidates failed SVD for column {col_idx}. "
                    f"N_val={N_val} is very small; using n_f={best_n_freq}, alpha={best_alpha}."
                )
            self.alphas_.append(best_alpha)
            self.n_frequencies_.append(best_n_freq)
            
        # Build X_list of DatasetObj for covariance computation (excluding all-NaN columns)
        valid_cols = []
        X_list = []
        for col_idx in range(X_data.shape[1]):
            col_data = X_data[:, col_idx]
            valid_mask = ~np.isnan(col_data) & ~np.isnan(self.timestamps_)
            if np.any(valid_mask):
                X_list.append(DatasetObj(self.timestamps_, col_data))
                valid_cols.append(col_idx)
            else:
                import warnings
                warnings.warn(f"Column {col_idx} is all NaN or has no valid timestamps, skipping from covariance compensation.")
            
        # If covariance compensation is requested, compute LDL^T
        if self.covariance_compensation:
            n_cols = X_data.shape[1]
            if len(X_list) > 0:
                lu_small, d_small, perm_small, valid_idx_comp = covariance_compensation(X_list, device=self.device)
                
                # Expand to full size; apply inverse permutation for correct column alignment
                self.lu_ = np.eye(n_cols)
                self.d_ = np.eye(n_cols)
                self.perm_ = np.arange(n_cols)
                
                n_valid = len(valid_cols)
                # valid_idx_comp references the doubled (real+imag) space. Extract the real part indices.
                real_indices = [i for i, v in enumerate(valid_idx_comp) if v < n_valid]
                
                if len(real_indices) > 0:
                    lu_real = lu_small[np.ix_(real_indices, real_indices)]
                    d_real = d_small[np.ix_(real_indices, real_indices)]
                    perm_real = np.argsort(np.argsort(perm_small[real_indices]))
                    
                    actual_valid_cols = [valid_cols[valid_idx_comp[i]] for i in real_indices]
                    n_small = len(actual_valid_cols)
                    inv_perm = np.argsort(perm_real)
                    
                    for i in range(n_small):
                        full_i = actual_valid_cols[i]
                        self.perm_[full_i] = actual_valid_cols[perm_real[i]]
                        self.d_[full_i, full_i] = d_real[inv_perm[i], inv_perm[i]]
                        for j in range(n_small):
                            full_j = actual_valid_cols[j]
                            self.lu_[full_i, full_j] = lu_real[inv_perm[i], inv_perm[j]]
            else:
                self.lu_ = np.eye(n_cols)
                self.d_ = np.eye(n_cols)
                self.perm_ = np.arange(n_cols)
            
        return self

    def transform(self, X, timestamps=None, stochastic=False, stochastic_scale=1.0):
        """
        Transforms X by infilling NaNs using the fitted NUDFT-based smooth reconstruction.
        Supports stochastic posterior sampling representing imputation uncertainty.
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ['alphas_', 'n_frequencies_', 'timestamps_'])
        from nufi.kernels.torch_kernels import solve_tikhonov_nudft
        
        if isinstance(X, pd.DataFrame):
            if timestamps is None:
                timestamps = X.index.to_numpy()
            X_data = X.to_numpy().copy()
        else:
            X_data = np.array(X).copy()
            if timestamps is None:
                timestamps = np.arange(len(X_data), dtype=np.float64)
                
        t_timestamps = np.array(timestamps, dtype=np.float64)
        dev = get_device(self.device)
        
        infilled_data = np.zeros_like(X_data)
        # Note: these dictionaries store per-column full-length arrays and may be large for many-column datasets.
        # Use locally scoped variables; DO NOT attach sample-sized arrays to `self` during transform.
        local_reconstructed = {}
        

        from sklearn.utils import check_random_state
        rng = check_random_state(self.random_state)
        
        for col_idx in range(X_data.shape[1]):
            col_data = X_data[:, col_idx]
            valid_mask = ~np.isnan(col_data) & ~np.isnan(t_timestamps)
            v_timestamps = t_timestamps[valid_mask]
            v_data = col_data[valid_mask]
            
            if len(v_data) == 0:
                infilled_data[:, col_idx] = col_data
                continue
                
            # Use fitted GCV parameters for this column
            alpha = self.alphas_[col_idx] if col_idx < len(self.alphas_) else (self.alpha if isinstance(self.alpha, (int, float)) else 1e-4)
            n_f = self.n_frequencies_[col_idx] if col_idx < len(self.n_frequencies_) else (self.n_frequencies if isinstance(self.n_frequencies, int) else len(X_data))
            
            v_timestamps, v_data, nyquist_frequency = self._sort_and_compute_nyquist(v_timestamps, v_data)
            f_k = np.linspace(0, nyquist_frequency, n_f)
            
            # Solve regularized system for continuous NUDFT spectrum F
            try:
                F = solve_tikhonov_nudft(
                    v_timestamps, v_data, f_k, alpha, 
                    solver=self.solver, max_iter=self.max_iter, tol=self.tol, device=self.device
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"NUDFT solver failed for column {col_idx} with alpha={alpha}, n_f={n_f}: {e}. "
                    f"Consider using a larger alpha or more observations."
                )
            # Reconstruct the signal at all timestamps
            t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
            t_times = torch.tensor(t_timestamps, dtype=torch.float64, device=dev)
            
            exponent = 2.0j * np.pi * t_times.unsqueeze(1) * t_f_k.unsqueeze(0)
            
            # The "Rug Sweep": We sum the complex exponentials and drop the imaginary 
            # noise because our target physical signal is purely real.
            # (V3 Optimization: Rewrite the solver matrix A to use purely real Sines/Cosines)
            reconstructed = torch.real(torch.sum(F.unsqueeze(0) * torch.exp(exponent), dim=1))
            
            local_reconstructed[col_idx] = reconstructed.cpu().numpy()
            
        # --- Apply Full LDL^T Covariance Compensation ---
        n_samples, n_cols = X_data.shape
        residual_stds = np.zeros(n_cols)
        
        # Calculate base residuals for each column using local_reconstructed
        for col_idx in range(n_cols):
            if col_idx not in local_reconstructed:
                residual_stds[col_idx] = 0.1
                continue
                
            nan_mask = np.isnan(X_data[:, col_idx])
            obs_mask = ~nan_mask
            if np.any(obs_mask):
                residual = X_data[obs_mask, col_idx] - local_reconstructed[col_idx][obs_mask]
                r_std = np.std(residual) if len(residual) > 1 else 0.1
                residual_stds[col_idx] = r_std if not (np.isnan(r_std) or r_std == 0) else 0.1
            else:
                residual_stds[col_idx] = 0.1

        if stochastic:
            # 1. Generate UN-SCALED standard normal noise
            eta = rng.normal(0, 1.0, size=(n_samples, n_cols))
            
            if self.covariance_compensation and self.lu_ is not None and self.d_ is not None and self.perm_ is not None:
                # Reconstruct the transformation matrix T
                L = self.lu_
                sqrt_D = np.diag(np.sqrt(np.abs(np.diag(self.d_))))
                P = np.eye(n_cols)[self.perm_]
                T = P @ L @ sqrt_D
                
                # 2. Apply structural correlation FIRST
                Z = eta @ T.T
                
                # 3. Normalize variances back to 1.0, THEN apply marginal scaling
                Z_vars = np.diag(T @ T.T)
                Z_vars[Z_vars == 0] = 1.0
                noise_correlated = (Z / np.sqrt(Z_vars)) * residual_stds * stochastic_scale
            else:
                noise_correlated = eta * residual_stds * stochastic_scale
        else:
            noise_correlated = np.zeros((n_samples, n_cols))

        # Fill NaNs with the reconstructed signal + correlated noise
        for col_idx in range(n_cols):
            if col_idx not in local_reconstructed:
                infilled_data[:, col_idx] = X_data[:, col_idx]
                continue
                
            nan_mask = np.isnan(X_data[:, col_idx])
            imputed_vals = local_reconstructed[col_idx][nan_mask] + noise_correlated[nan_mask, col_idx]
            X_data[nan_mask, col_idx] = imputed_vals
                    
            infilled_data[:, col_idx] = X_data[:, col_idx]
            
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(infilled_data, index=X.index, columns=X.columns)
        return infilled_data
