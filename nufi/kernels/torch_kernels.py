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

def compute_ND_NUDFT(X_list, device=None):
    """
    Computes the 1D Non-Uniform Discrete Fourier Transform (NUDFT) for each
    multidimensional signal in X_list using PyTorch acceleration.
    Gracefully handles NaNs by masking them out.
    """
    dev = get_device(device)
    results = []

    for X in X_list:
        # X should have X.timestamps and X.data
        timestamps = np.array(X.timestamps, dtype=np.float64)
        data = np.array(X.data, dtype=np.float64)

        # Mask out NaN values
        valid_mask = ~np.isnan(data) & ~np.isnan(timestamps)
        if not np.any(valid_mask):
            # All NaNs, return zeros
            results.append(torch.zeros(len(data), dtype=torch.complex128, device=dev))
            continue

        v_timestamps = timestamps[valid_mask]
        v_data = data[valid_mask]

        N = len(data)
        # Estimate Nyquist frequency from median/min sampling interval
        if len(v_timestamps) > 1:
            p_n = np.diff(v_timestamps)
            min_p = np.nanmin(p_n) if len(p_n) > 0 and np.nanmin(p_n) > 0 else 1.0
            nyquist_frequency = 0.5 / max(min_p, 1e-12)
        else:
            nyquist_frequency = 1.0

        f_k = torch.linspace(0, nyquist_frequency, N, dtype=torch.float64, device=dev)

        # Standard NUDFT: A[n,k] = exp(-2πi * t_n * f_k), then sum over n
        t_timestamps = torch.tensor(v_timestamps, dtype=torch.float64, device=dev)
        t_data_all = torch.tensor(v_data, dtype=torch.float64, device=dev)
        exponent = -2.0j * np.pi * t_timestamps.unsqueeze(1) * f_k.unsqueeze(0)
        summation = torch.sum(t_data_all.to(torch.complex128).unsqueeze(1) * torch.exp(exponent), dim=0)
        
        results.append(summation)

    return results

def compute_Fast_ND_NUDFT(X_list, device=None):
    """
    Performs Fast Non-Uniform DFT by interpolating onto a uniform grid 
    and computing FFT using PyTorch rfft/fft.
    Gracefully handles NaNs during interpolation.
    """
    dev = get_device(device)
    results = []

    for X in X_list:
        timestamps = np.array(X.timestamps, dtype=np.float64)
        data = np.array(X.data, dtype=np.float64)

        # Handle NaNs in data by linear interpolation/forward-fill
        valid_mask = ~np.isnan(data) & ~np.isnan(timestamps)
        if not np.any(valid_mask):
            results.append(torch.zeros(len(data), dtype=torch.complex128, device=dev))
            continue

        v_timestamps = timestamps[valid_mask]
        v_data = data[valid_mask]

        N = len(data)
        # Generate uniform grid using min/max of valid timestamps
        t_min, t_max = np.min(v_timestamps), np.max(v_timestamps)
        uniform_grid = np.linspace(t_min, t_max, N)
        # Interpolate onto uniform grid
        uniform_data = np.interp(uniform_grid, v_timestamps, v_data)

        # Compute FFT using PyTorch
        t_uniform_data = torch.tensor(uniform_data, dtype=torch.float64, device=dev)
        fft_result = torch.fft.fft(t_uniform_data)

        results.append(fft_result)

    return results

def covariance_compensation(X_list, device=None):
    """
    Processes the X_list, computes the multi-dimensional NUDFT, calculates
    the covariance matrix, and performs LDLᵀ decomposition.
    """
    dev = get_device(device)
    
    # Step 1: Compute NUDFT results
    X_k_result = compute_ND_NUDFT(X_list, device=dev)

    # Step 2: Flatten & stack the data (preserving phase information)
    # Move to CPU for covariance and LDL^T since scipy/pandas functions are optimized there
    flat_data = []
    for tensor in X_k_result:
        arr = tensor.cpu().numpy()
        flat_data.append(np.concatenate([arr.real, arr.imag]))

    flat_data = np.array(flat_data).T # Shape: samples x dimensions

    # Step 3: Compute covariance matrix
    # Handle any potential remaining NaNs just in case
    import pandas as pd
    df = pd.DataFrame(flat_data)
    covariance_matrix = df.cov().to_numpy()

    if np.any(np.isnan(covariance_matrix)):
        import warnings
        warnings.warn("Covariance matrix contains NaN entries; degenerate columns detected.")
    
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0)

    # Step 4: Perform LDL^T decomposition
    # LDL^T factorizes A = P * L * D * L^T
    lu, d, perm = scipy.linalg.ldl(covariance_matrix)

    return lu, d, perm

def solve_cg(A, b, alpha, max_iter=100, tol=1e-5):
    """
    Solves the regularized system (A^H A + alpha * I) F = b using the 
    iterative Conjugate Gradient method for complex vectors.
    """
    dev = A.device
    M = A.shape[1]
    x = torch.zeros(M, dtype=torch.complex128, device=dev)
    
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
        alpha_cg = rsold / denom
        x = x + alpha_cg * p
        r = r - alpha_cg * Hp
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
    Supports 'direct' SVD/normal-equation solvers or 'cg' Conjugate Gradient.
    """
    if alpha <= 0:
        raise ValueError(f"Regularization parameter alpha must be positive, got {alpha}")

    dev = get_device(device)
    t_timestamps = torch.tensor(timestamps, dtype=torch.float64, device=dev)
    t_data = torch.tensor(data, dtype=torch.float64, device=dev)
    t_f_k = torch.tensor(f_k, dtype=torch.float64, device=dev)
    
    # Build Fourier mapping matrix A: shape (N, M)
    # A_nk = exp(2*pi*i * f_k * t_n)
    exponent = 2.0j * np.pi * t_timestamps.unsqueeze(1) * t_f_k.unsqueeze(0)
    A = torch.exp(exponent)
    
    # Right-hand side b = A^H y
    b = torch.matmul(A.adjoint(), t_data.to(torch.complex128))
    
    if solver == 'cg':
        F = solve_cg(A, b, alpha, max_iter=max_iter, tol=tol)
    else:
        # Direct solver: (A^H A + alpha * I) F = A^H y
        M = A.shape[1]
        reg_matrix = A.adjoint() @ A + alpha * torch.eye(M, dtype=torch.complex128, device=dev)
        F = torch.linalg.solve(reg_matrix, b)
        
    return F
