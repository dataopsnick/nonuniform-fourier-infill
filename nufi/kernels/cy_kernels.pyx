# cy_kernels.pyx
import numpy as np
cimport numpy as cnp
from libc.math cimport sin, cos, M_PI, isnan
from cython.parallel import prange

cnp.import_array()

def compute_ND_NUDFT_cy(list X_list):
    """
    Cython optimized CPU NUDFT summation with multi-threading (prange).
    Gracefully ignores NaNs.
    """
    cdef int num_signals = len(X_list)
    cdef list results = []
    
    for idx in range(num_signals):
        X = X_list[idx]
        timestamps = np.ascontiguousarray(X.timestamps, dtype=np.float64)
        data = np.ascontiguousarray(X.data, dtype=np.float64)
        
        results.append(compute_single_NUDFT_cy(timestamps, data))
        
    return results

cdef compute_single_NUDFT_cy(double[:] timestamps, double[:] data):
    cdef int N = len(data)
    cdef int i, k, n   # <--- n is now properly typed as C int
    cdef double min_p = 1e9
    cdef double diff
    
    # Precompute min interval (excluding NaNs) without allocating throwaway arrays
    for i in range(N - 1):
        if not isnan(timestamps[i]) and not isnan(timestamps[i+1]):
            diff = timestamps[i+1] - timestamps[i]
            # Ensure absolute difference in case timestamps aren't strictly sorted
            if diff < 0:
                diff = -diff
            if diff > 0 and diff < min_p:
                min_p = diff
                
    if min_p == 1e9:
        min_p = 1.0
        
    cdef double max_sampling_rate = 1.0 / min_p
    cdef double nyquist_frequency = max_sampling_rate / 2.0
    cdef double[:] f_k = np.linspace(0, nyquist_frequency, N)
    
    # Pre-allocate output complex array
    cdef double complex[:] summation = np.zeros(N, dtype=np.complex128)
    
    cdef double p_val, f_val, d_val, angle
    
    # Multi-threaded computation over frequency bins
    for k in prange(N, nogil=True):
        f_val = f_k[k]
        for n in range(N):
            d_val = data[n]
            p_val = timestamps[n] # absolute time for analysis
            
            # Must guard against BOTH data NaNs and timestamp NaNs
            if not isnan(d_val) and not isnan(p_val):
                angle = -2.0 * M_PI * p_val * f_val
                summation[k] += d_val * (cos(angle) + 1j * sin(angle))
            
    return np.asarray(summation)