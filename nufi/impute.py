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
                 n_frequencies=None, alpha=1e-4, solver='direct', max_iter=100, tol=1e-5, random_state=None,
                 mnar_strategy=None, delta_adjustment=0.0):
        self.method = method
        self.device = device
        self.covariance_compensation = covariance_compensation
        self.n_frequencies = n_frequencies
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.mnar_strategy = mnar_strategy
        self.delta_adjustment = delta_adjustment
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
            random_state=self.random_state,
            mnar_strategy=self.mnar_strategy,
            delta_adjustment=self.delta_adjustment
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

    def transform(self, X, timestamps=None, stochastic=False, stochastic_scale=1.0,
                  n_imputations=1, mnar_strategy=None, delta_adjustment=None):
        """
        Transforms X by infilling NaNs using the fitted NUDFT-based smooth
        reconstruction. Supports stochastic posterior sampling.

        MNAR extensions (Tasks 46/49):
        - mnar_strategy / delta_adjustment: Pattern Mixture Model (PMM)
          adjustments for MNAR imputation ("pmm_shift", "pmm_slope",
          "spectral_dampening"); falls back to the __init__ values when not
          provided. Observed data are always strictly invariant.
        - n_imputations: when > 1 (requires stochastic=True), generates M
          correlated posterior realizations in a single vectorized pass:
          a 3D ndarray (M, n_samples, n_cols) for array input, or a list of
          M DataFrames for DataFrame input.
        """
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, ["alphas_", "n_frequencies_", "timestamps_"])
        from nufi.kernels.torch_kernels import solve_tikhonov_nudft

        if not isinstance(n_imputations, (int, np.integer)) or n_imputations < 1:
            raise ValueError(
                "n_imputations must be a positive integer, got %r"
                % (n_imputations,)
            )
        if n_imputations > 1 and not stochastic:
            raise ValueError(
                "n_imputations > 1 requires stochastic=True: multiple "
                "imputation draws are posterior realizations and are "
                "meaningless without stochastic noise."
            )

        if mnar_strategy is not None:
            strategy = mnar_strategy
        else:
            strategy = self.mnar_strategy
        if delta_adjustment is not None:
            delta = delta_adjustment
        else:
            delta = self.delta_adjustment
        strategy = validate_mnar_strategy(strategy)

        if isinstance(X, pd.DataFrame):
            if timestamps is None:
                timestamps = X.index.to_numpy()
            X_data = X.to_numpy().copy()
            column_names = list(X.columns)
        else:
            X_data = np.array(X).copy()
            if timestamps is None:
                timestamps = np.arange(len(X_data), dtype=np.float64)
            column_names = None

        t_timestamps = np.array(timestamps, dtype=np.float64)
        n_samples, n_cols = X_data.shape
        X_orig = X_data.copy()
        dev = get_device(self.device)

        # Per-column deterministic reconstructions and solved spectra, also
        # exposed on the imputer (reconstructed_, coefficients_, frequencies_)
        # so downstream RBMI / tipping-point / diagnostics reuse one solve.
        local_reconstructed = {}
        local_coefficients = {}
        local_frequencies = {}

        from sklearn.utils import check_random_state
        rng = check_random_state(self.random_state)

        for col_idx in range(n_cols):
            col_data = X_orig[:, col_idx]
            valid_mask = ~np.isnan(col_data) & ~np.isnan(t_timestamps)
            v_timestamps = t_timestamps[valid_mask]
            v_data = col_data[valid_mask]

            if len(v_data) == 0:
                continue

            if col_idx < len(self.alphas_):
                alpha = self.alphas_[col_idx]
            elif isinstance(self.alpha, (int, float)):
                alpha = self.alpha
            else:
                alpha = 1e-4
            if col_idx < len(self.n_frequencies_):
                n_f = self.n_frequencies_[col_idx]
            elif isinstance(self.n_frequencies, int):
                n_f = self.n_frequencies
            else:
                n_f = n_samples

            v_ts, v_dat, nyq = self._sort_and_compute_nyquist(v_timestamps, v_data)
            f_k = np.linspace(0, nyq, n_f)

            # Solve regularized system for continuous NUDFT spectrum F
            try:
                F = solve_tikhonov_nudft(
                    v_ts, v_dat, f_k, alpha,
                    solver=self.solver, max_iter=self.max_iter,
                    tol=self.tol, device=self.device
                )
            except RuntimeError as e:
                raise RuntimeError(
                    "NUDFT solver failed for column %d with alpha=%s, "
                    "n_f=%d: %s. Consider a larger alpha or more "
                    "observations." % (col_idx, alpha, n_f, e)
                )

            if strategy == "spectral_dampening":
                delta_col, _ = _resolve_delta_for_column(
                    delta, col_idx, column_names, n_cols=n_cols,
                    allow_elementwise=False
                )
                if n_f > 1:
                    damp = np.exp(-abs(delta_col) * (f_k / f_k.max()) ** 2)
                else:
                    damp = np.ones(n_f)
                F = F * torch.tensor(damp, dtype=torch.float64, device=F.device)

            local_coefficients[col_idx] = F.detach().cpu().numpy()
            local_frequencies[col_idx] = f_k
            t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
            t_times = torch.tensor(t_timestamps, dtype=torch.float64, device=dev)
            exponent = 2.0j * np.pi * t_times.unsqueeze(1) * t_f_k.unsqueeze(0)
            reconstructed = torch.real(
                torch.sum(F.unsqueeze(0) * torch.exp(exponent), dim=1)
            )
            local_reconstructed[col_idx] = reconstructed.cpu().numpy()


        # Marginal residual scales (LDL^T compensation inputs)
        residual_stds = np.zeros(n_cols)
        for col_idx in range(n_cols):
            if col_idx not in local_reconstructed:
                residual_stds[col_idx] = 0.1
                continue
            nan_mask = np.isnan(X_orig[:, col_idx])
            obs_mask = ~nan_mask
            if np.any(obs_mask):
                residual = (
                    X_orig[obs_mask, col_idx]
                    - local_reconstructed[col_idx][obs_mask]
                )
                r_std = np.std(residual) if len(residual) > 1 else 0.1
                ok = not (np.isnan(r_std) or r_std == 0)
                residual_stds[col_idx] = r_std if ok else 0.1
            else:
                residual_stds[col_idx] = 0.1

        # Persist reusable fitted state for RBMI / tipping-point / diagnostics
        self.reconstructed_ = dict(local_reconstructed)
        self.coefficients_ = dict(local_coefficients)
        self.frequencies_ = dict(local_frequencies)
        self.residual_stds_ = residual_stds
        self.noise_transform_ = self._build_noise_transform()

        # Sample correlated posterior noise (vectorized over M draws)
        if stochastic:
            if n_imputations > 1:
                eta_shape = (n_imputations, n_samples, n_cols)
            else:
                eta_shape = (n_samples, n_cols)
            eta = rng.normal(0, 1.0, size=eta_shape)
            T = self.noise_transform_
            if T is not None:
                Z_vars = np.diag(T @ T.T)
                Z_vars[Z_vars == 0] = 1.0
                if n_imputations > 1:
                    Z = np.einsum("msc,dc->msd", eta, T)
                else:
                    Z = eta @ T.T
                noise_draws = (Z / np.sqrt(Z_vars)) * residual_stds
                noise_draws = noise_draws * stochastic_scale
            else:
                noise_draws = eta * residual_stds * stochastic_scale
        else:
            if n_imputations > 1:
                noise_draws = np.zeros((n_imputations, n_samples, n_cols))
            else:
                noise_draws = np.zeros((n_samples, n_cols))

        # Resolve per-column PMM delta offsets (Task 46)
        mnar_offsets = {}
        for col_idx in range(n_cols):
            if col_idx not in local_reconstructed:
                continue
            nan_mask = np.isnan(X_orig[:, col_idx])
            if strategy == "pmm_shift":
                delta_col, elem = _resolve_delta_for_column(
                    delta, col_idx, column_names, n_cols=n_cols,
                    n_samples=n_samples, allow_elementwise=True
                )
                offsets = np.zeros(n_samples)
                if elem is not None:
                    offsets[nan_mask] = elem[:, col_idx][nan_mask]
                else:
                    offsets[nan_mask] = delta_col
            elif strategy == "pmm_slope":
                delta_col, _ = _resolve_delta_for_column(
                    delta, col_idx, column_names, n_cols=n_cols,
                    allow_elementwise=False
                )
                offsets = compute_mnar_offsets(
                    "pmm_slope", delta_col, t_timestamps, nan_mask
                )
            else:
                offsets = np.zeros(n_samples)
            mnar_offsets[col_idx] = offsets

        def _fill_draw(noise_matrix):
            """Fill NaNs; observed data invariant under every MNAR strategy."""
            out = X_orig.copy()
            for col_idx in range(n_cols):
                if col_idx not in local_reconstructed:
                    continue
                nan_mask = np.isnan(out[:, col_idx])
                if not np.any(nan_mask):
                    continue
                out[nan_mask, col_idx] = (
                    local_reconstructed[col_idx][nan_mask]
                    + mnar_offsets[col_idx][nan_mask]
                    + noise_matrix[nan_mask, col_idx]
                )
            return out

        if n_imputations == 1:
            infilled_data = _fill_draw(noise_draws)
            if isinstance(X, pd.DataFrame):
                return pd.DataFrame(
                    infilled_data, index=X.index, columns=X.columns
                )
            return infilled_data

        draws = [_fill_draw(noise_draws[m]) for m in range(n_imputations)]
        if isinstance(X, pd.DataFrame):
            return [
                pd.DataFrame(d, index=X.index, columns=X.columns)
                for d in draws
            ]
        return np.stack(draws, axis=0)

    def _build_noise_transform(self):
        """Rebuilds the LDL^T spatial transform T = P L sqrt(D)."""
        if (
            self.covariance_compensation
            and self.lu_ is not None
            and self.d_ is not None
            and self.perm_ is not None
        ):
            L = self.lu_
            sqrt_D = np.diag(np.sqrt(np.abs(np.diag(self.d_))))
            P = np.eye(L.shape[0])[self.perm_]
            return P @ L @ sqrt_D
        return None

    def sample_posterior_noise(self, n_samples, n_imputations=1,
                               stochastic_scale=1.0, rng=None):
        """
        Draws correlated posterior noise realizations from the stored LDL^T
        transform and residual scales (populated by the last transform call).
        Allows RBMI and tipping-point workflows to reuse a single spectral
        solve across M draws. Returns shape (n_samples, n_cols) when
        n_imputations == 1, else (n_imputations, n_samples, n_cols).
        """
        from sklearn.utils.validation import check_is_fitted
        from sklearn.utils import check_random_state
        check_is_fitted(self, ["residual_stds_", "noise_transform_"])
        if not isinstance(n_imputations, (int, np.integer)) or n_imputations < 1:
            raise ValueError(
                "n_imputations must be a positive integer, got %r"
                % (n_imputations,)
            )
        if rng is None:
            rng = check_random_state(self.random_state)
        residual_stds = np.asarray(self.residual_stds_, dtype=np.float64)
        n_cols = residual_stds.size
        if n_imputations > 1:
            shape = (n_imputations, n_samples, n_cols)
        else:
            shape = (n_samples, n_cols)
        eta = rng.normal(0, 1.0, size=shape)
        T = self.noise_transform_
        if T is not None:
            Z_vars = np.diag(T @ T.T)
            Z_vars[Z_vars == 0] = 1.0
            if n_imputations > 1:
                Z = np.einsum("msc,dc->msd", eta, T)
            else:
                Z = eta @ T.T
            noise = (Z / np.sqrt(Z_vars)) * residual_stds * stochastic_scale
        else:
            noise = eta * residual_stds * stochastic_scale
        return noise


def validate_mnar_strategy(mnar_strategy):
    """Validates the MNAR pattern-mixture strategy parameter (Task 46)."""
    allowed = (None, "pmm_shift", "pmm_slope", "spectral_dampening")
    if mnar_strategy not in allowed:
        raise ValueError(
            "mnar_strategy %r is not supported. Expected one of: "
            "None, pmm_shift, pmm_slope, spectral_dampening."
            % (mnar_strategy,)
        )
    return mnar_strategy


def compute_mnar_offsets(mnar_strategy, delta_value, t_timestamps, nan_mask):
    """
    Computes Pattern Mixture Model additive offsets for one column (Task 46).

    - "pmm_shift": constant delta_value at every missing position.
    - "pmm_slope": delta_value * max(0, t - t_anchor), where t_anchor is
      the last observed timestamp before each contiguous missing run,
      preserving C^0 continuity at the dropout boundary.
    - otherwise: zeros ("spectral_dampening" acts in the spectral domain).
    """
    nan_mask = np.asarray(nan_mask, dtype=bool)
    t_timestamps = np.asarray(t_timestamps, dtype=np.float64)
    offsets = np.zeros(nan_mask.size, dtype=np.float64)
    if mnar_strategy == "pmm_shift":
        offsets[nan_mask] = delta_value
        return offsets
    if mnar_strategy == "pmm_slope":
        nan_idx = np.where(nan_mask)[0]
        obs_idx = np.where(~nan_mask)[0]
        if nan_idx.size == 0 or obs_idx.size == 0:
            return offsets
        pos = np.searchsorted(obs_idx, nan_idx, side="right") - 1
        valid = pos >= 0
        t_anchor = t_timestamps[obs_idx[pos[valid]]]
        t_missing = t_timestamps[nan_idx[valid]]
        offsets[nan_idx[valid]] = delta_value * np.maximum(
            0.0, t_missing - t_anchor
        )
        return offsets
    return offsets


def _resolve_delta_for_column(delta_adjustment, col_idx, column_names=None,
                              n_cols=None, n_samples=None,
                              allow_elementwise=False):
    """
    Resolves a delta_adjustment (float | dict | ndarray) to a scalar for one
    column. Returns (delta_scalar, elementwise_2d_or_None). A 2-D ndarray is
    only allowed when allow_elementwise is True (i.e. "pmm_shift").
    """
    delta = delta_adjustment
    if delta is None:
        return 0.0, None
    if isinstance(delta, dict):
        if column_names is not None:
            name = column_names[col_idx] if col_idx < len(column_names) else None
            if name is not None and name in delta:
                return float(delta[name]), None
        if col_idx in delta:
            return float(delta[col_idx]), None
        return 0.0, None
    if np.isscalar(delta):
        return float(delta), None
    arr = np.asarray(delta, dtype=np.float64)
    if arr.ndim == 1:
        if n_cols is not None and arr.size != n_cols:
            raise ValueError(
                "1-D delta_adjustment has length %d; expected n_cols=%d."
                % (arr.size, n_cols)
            )
        if col_idx < arr.size:
            return float(arr[col_idx]), None
        return 0.0, None
    if arr.ndim == 2:
        if not allow_elementwise:
            raise ValueError(
                "2-D (element-wise) delta_adjustment is only supported "
                "with mnar_strategy=pmm_shift."
            )
        if n_cols is not None and arr.shape[1] != n_cols:
            raise ValueError(
                "2-D delta_adjustment has %d columns; expected n_cols=%d."
                % (arr.shape[1], n_cols)
            )
        if n_samples is not None and arr.shape[0] != n_samples:
            raise ValueError(
                "2-D delta_adjustment has %d rows; expected n_samples=%d."
                % (arr.shape[0], n_samples)
            )
        return 0.0, arr
    raise ValueError(
        "delta_adjustment must be a float, dict, or ndarray; got %s."
        % type(delta).__name__
    )


def evaluate_fourier_reconstruction(coefficients, frequencies, times):
    """
    Evaluates a solved NUFI spectrum at arbitrary timestamps (synthesis
    convention): x(t) = Re(sum_k F_k * exp(2*pi*i*f_k*t)).
    """
    F = np.asarray(coefficients)
    f_k = np.asarray(frequencies, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    exponent = 2.0j * np.pi * t[:, None] * f_k[None, :]
    return np.real(np.sum(F[None, :] * np.exp(exponent), axis=1))


def evaluate_fourier_derivative(coefficients, frequencies, times):
    """
    Analytical first derivative of the NUFI reconstruction (used for
    Diggle-Kenward MNAR diagnostics):
    dx/dt = Re(sum_k (2*pi*i*f_k) * F_k * exp(2*pi*i*f_k*t)).
    """
    F = np.asarray(coefficients)
    f_k = np.asarray(frequencies, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    exponent = 2.0j * np.pi * t[:, None] * f_k[None, :]
    deriv_coeffs = (2.0j * np.pi * f_k) * F
    return np.real(np.sum(deriv_coeffs[None, :] * np.exp(exponent), axis=1))


def pool_rubins_rules(estimates, variances, confidence_level=0.95, n_obs=None):
    """
    Pools M point estimates and within-imputation variances using Rubins
    Rules (Rubin, 1987), with the Barnard & Rubin (1999) small-sample
    degrees-of-freedom adjustment when n_obs is provided. The dof and FMI
    formulas follow the reference implementation in the R mice package
    (van Buuren & Groothuis-Oudshoorn, 2011).

    Returns a dict with: pooled_estimate (Qbar), within_variance (Ubar),
    between_variance (B), total_variance (T = Ubar + (1 + 1/M) B), dof, fmi,
    t_stat, p_value, ci_bounds, n_imputations, relative_increase (r).
    """
    from scipy import stats as _stats
    q = np.asarray(estimates, dtype=np.float64).ravel()
    u = np.asarray(variances, dtype=np.float64).ravel()
    if q.size != u.size:
        raise ValueError(
            "estimates and variances must have the same length; "
            "got %d and %d." % (q.size, u.size)
        )
    M = int(q.size)
    if M < 2:
        raise ValueError(
            "Rubins Rules pooling requires at least M >= 2 imputations."
        )
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(u)):
        raise ValueError("estimates and variances must be finite.")
    if np.any(u < 0):
        raise ValueError("variances must be non-negative.")
    if not (0.0 < float(confidence_level) < 1.0):
        raise ValueError(
            "confidence_level must be in (0, 1); got %s." % (confidence_level,)
        )

    q_bar = float(np.mean(q))
    u_bar = float(np.mean(u))
    b = float(np.var(q, ddof=1))
    total_var = u_bar + (1.0 + 1.0 / M) * b

    if total_var > 0.0:
        # Relative increase in variance due to nonresponse (Rubin 1987)
        if u_bar > 0.0:
            r = (1.0 + 1.0 / M) * b / u_bar
        else:
            r = np.inf
        lam = (1.0 + 1.0 / M) * b / total_var
        if lam > 0.0:
            dof_old = (M - 1) / (lam ** 2)
        else:
            dof_old = np.inf
        if n_obs is not None:
            dfcom = float(n_obs) - 1.0
            if np.isfinite(dfcom) and dfcom > 0.0 and 0.0 < lam < 1.0:
                # Barnard & Rubin (1999) adjusted df (mice::barnard.rubin)
                tmp = (1.0 - lam) * (1.0 + dfcom) * dfcom
                dof = (M - 1) * tmp / (
                    (dfcom + 3.0) * (M - 1) + (lam ** 2) * tmp
                )
            else:
                dof = dof_old
        else:
            dof = dof_old
        if np.isfinite(r):
            fmi = (r + 2.0 / (dof + 3.0)) / (1.0 + r)
        else:
            fmi = 1.0
        t_stat = q_bar / np.sqrt(total_var)
        p_value = float(2.0 * _stats.t.sf(abs(t_stat), dof))
        tcrit = float(
            _stats.t.ppf(1.0 - (1.0 - float(confidence_level)) / 2.0, dof)
        )
        half_width = tcrit * np.sqrt(total_var)
        ci_bounds = (float(q_bar - half_width), float(q_bar + half_width))
    else:
        # Degenerate: zero total variance (identical draws, zero variances)
        r = 0.0
        lam = 0.0
        dof = np.inf
        fmi = 0.0
        t_stat = np.nan
        p_value = np.nan
        ci_bounds = (q_bar, q_bar)

    return {
        "pooled_estimate": q_bar,
        "within_variance": u_bar,
        "between_variance": b,
        "total_variance": float(total_var),
        "dof": float(dof),
        "fmi": float(fmi),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_bounds": ci_bounds,
        "n_imputations": M,
        "relative_increase": float(r),
    }
