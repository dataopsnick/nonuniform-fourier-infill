# triton_kernels.py
import torch
import triton
import triton.language as tl

@triton.jit
def _nudft_reconstruct_kernel(
    t_ptr,              # [M] query timestamps
    f_ptr,              # [K] frequencies
    F_real_ptr,         # [K] real part of coefficients
    F_imag_ptr,         # [K] imaginary part of coefficients
    out_ptr,            # [M] output signal
    M,                  # Total number of timestamps
    K,                  # Total number of frequencies
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Computes Re(Sum_k (F_k * exp(2j * pi * t * f_k))) in O(1) auxiliary memory.
    """
    pid = tl.program_id(0)
    
    # 1. Identify which block of timestamps this thread block is responsible for
    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    m_mask = m_offsets < M
    
    # Load timestamps into SRAM
    t = tl.load(t_ptr + m_offsets, mask=m_mask, other=0.0)
    
    # Initialize the accumulator for the reconstructed signal in registers
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # 2. Tile over the frequency domain
    for k_idx in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_offsets = k_idx * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < K
        
        # Load frequencies and coefficients into SRAM
        f = tl.load(f_ptr + k_offsets, mask=k_mask, other=0.0)
        F_real = tl.load(F_real_ptr + k_offsets, mask=k_mask, other=0.0)
        F_imag = tl.load(F_imag_ptr + k_offsets, mask=k_mask, other=0.0)
        
        # Expand dimensions to trigger Triton's broadcast semantics for outer product
        # t: [BLOCK_SIZE_M, 1] | f: [1, BLOCK_SIZE_K]
        t_exp = tl.expand_dims(t, 1)
        f_exp = tl.expand_dims(f, 0)
        
        # Compute theta = 2 * pi * t * f
        theta = 6.283185307179586 * t_exp * f_exp
        
        # Compute trig functions directly in registers
        cos_theta = tl.math.cos(theta)
        sin_theta = tl.math.sin(theta)
        
        F_real_exp = tl.expand_dims(F_real, 0)
        F_imag_exp = tl.expand_dims(F_imag, 0)
        
        # Re(F * e^{i*theta}) = A*cos(theta) - B*sin(theta)
        real_part = (F_real_exp * cos_theta) - (F_imag_exp * sin_theta)
        
        # Sum across the frequency dimension (axis=1) and accumulate
        acc += tl.sum(real_part, axis=1)
        
    # 3. Write the fully accumulated block back to HBM
    tl.store(out_ptr + m_offsets, acc, mask=m_mask)

def fast_nudft_reconstruct(t: torch.Tensor, f: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """
    Python wrapper to launch the Triton kernel.
    """
    assert t.is_cuda and f.is_cuda and F.is_cuda
    
    # Cast to float32. For timestamps, always subtract t[0] before calling this 
    # to avoid catastrophic cancellation in fp32 arithmetic.
    t = t.to(torch.float32)
    f = f.to(torch.float32)
    F_real = F.real.to(torch.float32)
    F_imag = F.imag.to(torch.float32)
    
    M = t.shape[0]
    K = f.shape[0]
    out = torch.empty((M,), device=t.device, dtype=torch.float32)
    
    # Grid configuration: 1D grid over the timestamps
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = 256
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    
    _nudft_reconstruct_kernel[grid](
        t, f, F_real, F_imag, out,
        M, K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    return out