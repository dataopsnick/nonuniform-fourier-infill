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
                candidates = sorted(list(set(candidates)))
            else:
                candidates = [self.n_frequencies if self.n_frequencies is not None else N_val]
                
            best_n_freq = candidates[0]
            best_alpha = self.alpha if isinstance(self.alpha, (int, float)) else 1e-4
            best_gcv = float('inf')
            
            p_n = np.diff(v_timestamps)
            min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
            max_sampling_rate = 1.0 / min_p
            nyquist_frequency = max_sampling_rate / 2.0
            
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
                        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
                        y_complex = t_data.to(torch.complex128)
                        y_tilde = torch.matmul(U.adjoint(), y_complex)
                        y_norm_sq = torch.sum(torch.abs(y_complex) ** 2)
                    
                    score = compute_gcv_from_svd(S, y_tilde, y_norm_sq, opt_alpha, N_val)
                except RuntimeError as e:
                    import warnings
                    warnings.warn(f"SVD failed for column {col_idx}, n_f={n_f}: {e}. Skipping candidate.")
                    continue
                
                if score < best_gcv:
                    best_gcv = score
                    best_n_freq = n_f
                    best_alpha = opt_alpha
            
            if best_gcv == float('inf'):
                import warnings
                warnings.warn(
                    f"All GCV candidates failed SVD for column {col_idx}. "
                    f"Using fallback n_f={best_n_freq}, alpha={best_alpha}."
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
                lu_small, d_small, perm_small = covariance_compensation(X_list, device=self.device)
                
                # Expand to full size; apply inverse permutation for correct column alignment
                self.lu_ = np.eye(n_cols)
                self.d_ = np.eye(n_cols)
                inv_perm = np.argsort(perm_small)
                self.perm_ = np.arange(n_cols)
                
                # Map small matrices back to full size using perm_small mapping
                if d_small.ndim == 1:
                    for i, c_i in enumerate(valid_cols):
                        self.d_[valid_cols[perm_small[i]], valid_cols[perm_small[i]]] = d_small[i]
                else:
                    for i, c_i in enumerate(valid_cols):
                        for j, c_j in enumerate(valid_cols):
                            self.d_[valid_cols[perm_small[i]], valid_cols[perm_small[j]]] = d_small[i, j]
                
                for i, c_i in enumerate(valid_cols):
                    for j, c_j in enumerate(valid_cols):
                        self.lu_[valid_cols[perm_small[i]], valid_cols[perm_small[j]]] = lu_small[i, j]
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
        
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        
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
            
            p_n = np.diff(v_timestamps)
            min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
            max_sampling_rate = 1.0 / min_p
            nyquist_frequency = max_sampling_rate / 2.0
            f_k = np.linspace(0, nyquist_frequency, n_f)
            
            # Solve regularized system for continuous NUDFT spectrum F
            F = solve_tikhonov_nudft(
                v_timestamps, v_data, f_k, alpha, 
                solver=self.solver, max_iter=self.max_iter, tol=self.tol, device=self.device
            )
            
            # Reconstruct the signal at all timestamps
            t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
            t_times = torch.tensor(t_timestamps, dtype=torch.float64, device=dev)
            
            exponent = 2.0j * np.pi * t_times.unsqueeze(1) * t_f_k.unsqueeze(0)
            reconstructed = torch.real(torch.sum(F.unsqueeze(0) * torch.exp(exponent), dim=1))
            reconstructed_np = reconstructed.cpu().numpy()
            
            reconstructed_raw = reconstructed_np.copy()
            # If covariance compensation is computed, align the reconstructed scale
            cov_scale = 1.0
            if self.covariance_compensation and self.d_ is not None:
                cov_scale = np.sqrt(np.abs(np.diag(self.d_)[col_idx]))
                if cov_scale > 0:
                    reconstructed_np = reconstructed_np * cov_scale
            
            # Fill only the NaNs
            nan_mask = np.isnan(X_data[:, col_idx])
            if np.any(nan_mask):
                if stochastic:
                    # Compute residual standard deviation on observed values
                    obs_mask = ~nan_mask
                    if np.any(obs_mask):
                        # Use unscaled reconstruction so both terms are in the same space
                        residual = X_data[obs_mask, col_idx] - reconstructed_raw[obs_mask]
                        residual_std = np.std(residual) if len(residual) > 1 else 0.1
                        if np.isnan(residual_std) or residual_std == 0:
                            residual_std = 0.1
                    else:
                        residual_std = 0.1
                        
                    # Generate noise from posterior process scaled by uncertainty parameters
                    noise = rng.normal(0, stochastic_scale * residual_std, size=nan_mask.sum())
                    
                    if self.covariance_compensation and self.d_ is not None and cov_scale > 0:
                        noise = noise * cov_scale
                    
                    X_data[nan_mask, col_idx] = reconstructed_np[nan_mask] + noise
                else:
                    X_data[nan_mask, col_idx] = reconstructed_np[nan_mask]
                    
            infilled_data[:, col_idx] = X_data[:, col_idx]
            
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(infilled_data, index=X.index, columns=X.columns)
        return infilled_data
