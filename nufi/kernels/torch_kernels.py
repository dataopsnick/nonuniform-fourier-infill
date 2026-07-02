import torch
import numpy as np
import scipy.linalg

def get_device(device_str=None):
    """
    Selects the acceleration device (CUDA, MPS, or CPU).
    """
    if device_str is not None:
        return torch.device(device_str)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def compute_ND_NUDFT(X_list, device=None, nyquist_frequency=None):
    """
    Computes the 1D Non-Uniform Discrete Fourier Transform (NUDFT) for each
    multidimensional signal in X_list using PyTorch acceleration.
    Gracefully handles NaNs by masking them out.

    .. note::
        Uses the forward (analysis) NUDFT convention: A[n,k] = exp(-2πi * t_n * f_k).
        WARNING: `solve_tikhonov_nudft` uses the synthesis convention exp(+2πi*t_n*f_k).
        Conjugate coefficients before passing between these functions to avoid phase errors.
    """
    dev = get_device(device)
    results = []
    
    if nyquist_frequency is None and len(X_list) > 1:
        import warnings
        warnings.warn(
            "nyquist_frequency not provided; estimating per-signal Nyquist. "
            "This may produce inconsistent frequency grids across signals. "
            "Pass an explicit nyquist_frequency for multi-signal workflows."
        )

    for X in X_list:
        timestamps = np.array(X.timestamps, dtype=np.float64)
        data = np.array(X.data, dtype=np.float64)

        valid_mask = ~np.isnan(data) & ~np.isnan(timestamps)
        if not np.any(valid_mask):
            results.append(torch.zeros(len(data), dtype=torch.complex128, device=dev))
            continue

        v_timestamps = timestamps[valid_mask]
        v_data = data[valid_mask]

        N = len(data)
        if nyquist_frequency is None:
            if len(v_timestamps) > 1:
                sort_idx = np.argsort(v_timestamps)
                sorted_ts = v_timestamps[sort_idx]
                p_n = np.diff(sorted_ts)
                p_n = p_n[p_n > 0] 
                if len(p_n) > 0:
                    median_p = np.median(p_n)
                    estimated_nyquist = 0.5 / max(median_p, 1e-12)
                else:
                    estimated_nyquist = 1.0
            else:
                estimated_nyquist = 1.0
        else:
            estimated_nyquist = nyquist_frequency

        f_k = torch.linspace(0, estimated_nyquist, N, dtype=torch.float64, device=dev)
        t_timestamps = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
        t_data_all = torch.tensor(v_data, dtype=torch.float64, device=dev)

        MAX_MEM_N = 10_000
        if N > MAX_MEM_N:
            raise ValueError(
                f"N={N} exceeds MAX_MEM_N={MAX_MEM_N}. "
                f"Use compute_Fast_ND_NUDFT for large signals to avoid excessive memory consumption."
            )

        # Forward NUDFT: A[n,k] = exp(-2πi * t_n * f_k)
        exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
        summation = torch.sum(t_data_all.to(torch.complex128).unsqueeze(1) * torch.exp(exponent), dim=0)
        
        results.append(summation)

    return results

def compute_Fast_ND_NUDFT(X_list, device=None, nyquist_frequency=None):
    """
    Replication of FINUFFT's 'Gaussian Spreading' method
    
    Type-1 Fast Non-Uniform DFT using Gaussian Gridding.
    Achieves O(N log N) complexity while perfectly preserving C^infinity continuity 
    and preventing the spectral leakage caused by linear interpolation.

    .. note::
        The ``nyquist_frequency`` parameter is reserved for future use and is
        currently ignored by the Gaussian gridding algorithm.
    """
    dev = get_device(device)
    results = []
    
    _ = nyquist_frequency  # reserved; not yet wired into the gridding algorithm
    if len(X_list) > 1:
        import warnings
        warnings.warn("nyquist_frequency is not yet used by the Fast NUDFT algorithm. "
                      "Output frequency scaling depends solely on the number of valid points.")

    for X in X_list:
        timestamps = np.array(X.timestamps, dtype=np.float64)
        data = np.array(X.data, dtype=np.float64)

        valid_mask = ~np.isnan(data) & ~np.isnan(timestamps)
        v_timestamps = timestamps[valid_mask]
        v_data = data[valid_mask]

        N_full = len(data)
        N = len(v_data)
        if N == 0:
            results.append(torch.zeros(N_full, dtype=torch.complex128, device=dev))
            continue

        # 1. Coordinate Scaling
        # Scale timestamps to [0, 2*pi] to standardize the Gaussian spread
        t_min, t_max = np.min(v_timestamps), np.max(v_timestamps)
        if t_max <= t_min:
            import warnings
            warnings.warn("All non-NaN timestamps are identical; spectrum will be degenerate.")
        span = t_max - t_min if t_max > t_min else 1.0
        t_scaled = (v_timestamps - t_min) / span * (2 * np.pi)

        t_pt = torch.tensor(t_scaled, dtype=torch.float64, device=dev)
        y_pt = torch.tensor(v_data, dtype=torch.float64, device=dev).to(torch.complex128)

        # 2. Gridding Parameters
        M = max(int(2 * N), 2)  # 2x Oversampled grid
        sigma = 1.0 / np.sqrt(M) # Gaussian variance 
        W = 6                    # Spread window (update 6 adjacent grid points per sample)

        grid = torch.zeros(M, dtype=torch.complex128, device=dev)

        # 3. Gaussian Spreading (Scatter Add)
        # Find the nearest uniform grid index for every non-uniform timestamp
        grid_pos = (t_pt / (2 * np.pi)) * M
        grid_idx = torch.floor(grid_pos).to(torch.long)

        # Smear the amplitude across the +/- 3 neighboring grid points
        for w in range(-W//2 + 1, W//2 + 1):
            idx = (grid_idx + w) % M
            
            # Distance from the exact timestamp to the target grid point
            dist = grid_pos - (grid_idx + w)
            dist_rad = dist * (2 * np.pi / M) # Convert back to radians
            
            # Gaussian weight: exp(-x^2 / 2sigma^2)
            weight = torch.exp(- (dist_rad**2) / (2 * sigma**2))
            
            # Vectorized GPU accumulation
            grid.scatter_add_(0, idx, y_pt * weight)

        # 4. Standard Fast Fourier Transform
        F_grid = torch.fft.fft(grid)

        # 5. Apodization Deconvolution
        # Guard against N_full exceeding the FFT grid size M
        n_out = min(N_full, M)
        # Divide out the effect of the Gaussian smear in the frequency domain
        k = torch.arange(n_out, dtype=torch.float64, device=dev)
        # Use a sigma scaling that matches the computed sigma from M
        apodization = torch.exp((k**2) * (sigma**2) / 2.0)
        
        # Extract the positive frequencies, apply analytic correction and grid normalisation
        # To maintain length matching len(data) = N_full, we slice and compute up to N_full
        F = F_grid[:n_out] * apodization / M
        # Pad to N_full length if needed (higher frequencies are beyond resolution)
        if n_out < N_full:
            pad = torch.zeros(N_full - n_out, dtype=torch.complex128, device=dev)
            F = torch.cat([F, pad])        
        results.append(F)

    return results

def covariance_compensation(X_list, device=None):
    """
    Processes the X_list, computes the multi-dimensional NUDFT, calculates
    the covariance matrix, and performs LDLᵀ decomposition.
    """
    dev = get_device(device)
    
    # Step 1: Estimate a common Nyquist across all signals to ensure aligned frequency bins
    all_diffs = []
    for X in X_list:
        ts = np.array(X.timestamps, dtype=np.float64)
        v_ts = ts[~np.isnan(ts)]
        if len(v_ts) > 1:
            sorted_ts = np.sort(v_ts)
            diffs = np.diff(sorted_ts)
            all_diffs.extend(diffs[diffs > 0].tolist())
            
    if all_diffs:
        common_nyquist = 0.5 / max(np.median(all_diffs), 1e-12)
    else:
        common_nyquist = 1.0

    X_k_result = compute_ND_NUDFT(X_list, device=dev, nyquist_frequency=common_nyquist)

    # Step 2: Flatten & stack the data (preserving phase information)
    # Move to CPU for covariance and LDL^T since scipy/pandas functions are optimized there
    # Validate equal signal lengths before stacking
    lens = [len(tensor) for tensor in X_k_result]
    if len(set(lens)) > 1:
        raise ValueError(
            f"All signals must have the same length, got lengths {lens}. "
            f"Signals of different lengths cannot be covariance-compensated."
        )
        
    flat_data = []
    for tensor in X_k_result:
        arr = tensor.cpu().numpy()
        flat_data.append(np.concatenate([arr.real, arr.imag]))

    flat_data = np.array(flat_data).T # Shape: samples x dimensions
    # Step 3: Compute covariance matrix
    # Shape: (num_samples, num_dimensions)
    covariance_matrix = np.cov(flat_data, rowvar=False)

    # Detect degenerate columns on the diagonal of the covariance matrix
    diag = np.diag(covariance_matrix)
    diag_nan = np.isnan(diag)
    valid_idx = np.arange(covariance_matrix.shape[0]) # default: all valid
    
    if np.any(diag_nan):
        import warnings
        warnings.warn(f"Covariance matrix contains NaN on diagonal in {diag_nan.sum()} entries. Dropping degenerate columns.")
        # Drop degenerate real/imag pairs together: for each NaN diagonal entry at index k,
        # also drop its paired component (assuming real at even indices, imag at odd, or
        # N real then N imag layout).
        # For N-real-then-N-imag layout, paired index is (k + N) % (2*N).
        N = covariance_matrix.shape[0] // 2
        pair_mask = np.zeros(covariance_matrix.shape[0], dtype=bool)
        for k in np.where(diag_nan)[0]:
            pair_mask[k] = True
            pair_mask[(k + N) % (2 * N)] = True
        valid_idx = np.where(~pair_mask)[0]
        if len(valid_idx) == 0:
            raise ValueError("All columns are degenerate; cannot compute covariance compensation.")
        covariance_matrix = covariance_matrix[np.ix_(valid_idx, valid_idx)]

    # Step 3: LDL^T decomposition with progressive regularization fallback
    diag_mean = np.mean(np.diag(covariance_matrix))
    eps = max(1e-10 * max(diag_mean, 0.0), 1e-10)
    
    max_retries = 5
    for attempt in range(max_retries):
        cov_reg = covariance_matrix + eps * np.eye(covariance_matrix.shape[0])
        # Step 4: Perform LDL^T decomposition
        # LDL^T factorizes A = P * L * D * L^T
        try:
            lu, d, perm = scipy.linalg.ldl(cov_reg)
            break
        except Exception:
            if attempt == max_retries - 1:
                raise ValueError("LDL decomposition failed even after regularization; covariance matrix may be singular.")
            eps *= 10.0

    # valid_idx: indices into the full (pre-drop) covariance matrix.
    # Caller must map back to signal space: signal_idx = valid_idx // 2
    return lu, d, perm, valid_idx  

def solve_cg(A, b, alpha, max_iter=100, tol=1e-5):
    """
    Solves the regularized system (A^H A + alpha * I) F = b using the 
    iterative Conjugate Gradient method for complex vectors.

    .. note::
        H(v) = A^H A v + alpha v applies CG to the normal equations,
        which squares the condition number of A. For ill-conditioned A, prefer
        the augmented-system (direct) solver in solve_tikhonov_nudft.
    """
    dev = A.device
    M = A.shape[1]
    x = torch.zeros(M, dtype=torch.complex128, device=dev)

    # NOTE: H(v) = A^H A v + alpha v applies CG to the normal equations,
    # which squares the condition number of A. For ill-conditioned A, prefer
    # the augmented-system (direct) solver in solve_tikhonov_nudft.
    def H(v):
        return torch.matmul(A.adjoint(), torch.matmul(A, v)) + alpha * v
        
    r = b - H(x)
    p = r.clone()
    rsold = torch.sum(r.conj() * r).real
    
    if rsold < 1e-18:
        return x

    for i in range(max_iter):
        Hp = H(p)
        denom = torch.sum(p.conj() * Hp).real
        if denom < 1e-18:
            break
        step_size = rsold / denom
        x = x + step_size * p
        r = r - step_size * Hp
        rsnew = torch.sum(r.conj() * r).real
        if torch.sqrt(rsnew) < tol or rsnew < 1e-18:
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x

def compute_gcv_from_svd(s, y_tilde, y_norm_sq, alpha, N):
    """
    Computes Generalized Cross-Validation (GCV) score using singular values.
    """
    s_sq = s ** 2
    f = s_sq / (s_sq + alpha)
    tr_H = torch.sum(f)
    
    y_tilde_sq = torch.abs(y_tilde) ** 2
    res_norm_sq = torch.sum(((f - 1.0) ** 2) * y_tilde_sq) + (y_norm_sq - torch.sum(y_tilde_sq))
    
    denom = (1.0 - tr_H / N) ** 2
    if denom < 1e-8:
        return float('inf')
    return ((res_norm_sq / N) / denom).item()

def optimize_alpha_gcv(A, y, alphas=None, return_svd=False):
    """
    Finds the optimal Tikhonov regularization parameter alpha using GCV.
    """
    if alphas is None:
        alphas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    if len(alphas) == 0:
        raise ValueError("alphas must contain at least one candidate value.")
    
    N, M = A.shape
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    y_complex = y.to(torch.complex128)
    y_tilde = torch.matmul(U.adjoint(), y_complex)
    y_norm_sq = torch.sum(torch.abs(y_complex) ** 2)
    
    best_alpha = alphas[0]
    best_score = float('inf')
    for alpha in alphas:
        score = compute_gcv_from_svd(S, y_tilde, y_norm_sq, alpha, N)
        if score < best_score:
            best_score = score
            best_alpha = alpha

    if return_svd:
        return best_alpha, U, S, y_tilde, y_norm_sq
    return best_alpha

def solve_tikhonov_nudft(timestamps, data, f_k, alpha, solver='direct', max_iter=100, tol=1e-5, device=None):
    """
    Solves the continuous-time NUDFT coefficients using L2/Tikhonov regularization.
    
    .. note::
        WARNING: Uses the synthesis convention exp(+2πi*t_n*f_k). If you obtained 
        coefficients from `compute_ND_NUDFT`, you must conjugate them first.
    """
    if solver not in ('direct', 'cg'):
        raise ValueError(f"Unknown solver '{solver}'. Supported solvers: 'direct', 'cg'.")
        
    if alpha <= 0:
        raise ValueError(f"Regularization parameter alpha must be positive, got {alpha}")

    dev = get_device(device)
    # Validate inputs
    timestamps = np.asarray(timestamps, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)
    f_k = np.asarray(f_k, dtype=np.float64)
    
    valid_mask = ~np.isnan(timestamps) & ~np.isnan(data) & ~np.isnan(f_k)
    valid_mask &= ~np.isinf(timestamps) & ~np.isinf(data) & ~np.isinf(f_k)
    if not np.all(valid_mask):
        raise ValueError("Input timestamps, data, or f_k contain NaN/Inf values.")

    t_timestamps = torch.tensor(timestamps, dtype=torch.float64, device=dev)
    t_data = torch.tensor(data, dtype=torch.float64, device=dev)
    t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
    
    # Build Fourier mapping matrix A: shape (N, M)
    # A_nk = exp(2*pi*i * f_k * t_n)
    N, M = len(t_timestamps), len(t_f_k)
    MAX_ELEMENTS = 50_000_000  # ~800 MB for complex128
    
    # Guard includes augmented size [A; sqrt(alpha)*I]
    if (N + M) * M > MAX_ELEMENTS and solver != 'cg':
        raise ValueError(
            f"Augmented matrix shape ({N+M},{M}) has {(N+M)*M} elements; exceeds memory safety limit. "
            f"Use solver='cg' to avoid materializing the full matrix."
        )
        
    exponent = 2.0j * np.pi * t_timestamps.unsqueeze(1) * t_f_k.unsqueeze(0)
    A = torch.exp(exponent)
    
    # Right-hand side b = A^H y
    b = torch.matmul(A.adjoint(), t_data.to(torch.complex128))
    
    if solver == 'cg':
        F = solve_cg(A, b, alpha, max_iter=max_iter, tol=tol)
    else:
        # Direct solver via least-squares on augmented system [A; sqrt(alpha)*I] @ F = [y; 0]
        # This avoids squaring the condition number of A.
        M = A.shape[1]
        A_aug = torch.vstack([A, torch.sqrt(torch.tensor(alpha, dtype=torch.float64, device=dev)) * torch.eye(M, dtype=torch.complex128, device=dev)])
        b_aug = torch.cat([t_data.to(torch.complex128), torch.zeros(M, dtype=torch.complex128, device=dev)])
        F = torch.linalg.lstsq(A_aug, b_aug.unsqueeze(1)).solution.squeeze(1)
        
    return F