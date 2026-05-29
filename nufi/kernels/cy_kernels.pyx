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
    cdef int i, k
    cdef double[:] p_n = np.zeros(N - 1, dtype=np.float64)
    cdef double min_p = 1e9
    
    # Precompute time diffs and find min interval (excluding NaNs)
    for i in range(N - 1):
        if isnan(timestamps[i]) or isnan(timestamps[i+1]):
            p_n[i] = 0.0
        else:
            p_n[i] = timestamps[i+1] - timestamps[i]
            if p_n[i] > 0 and p_n[i] < min_p:
                min_p = p_n[i]
                
    if min_p == 1e9:
        min_p = 1.0
        
    cdef double max_sampling_rate = 1.0 / min_p
    cdef double nyquist_frequency = max_sampling_rate / 2.0
    cdef double[:] f_k = np.linspace(0, nyquist_frequency, N)
    
    # Pre-allocate output complex array
    cdef double complex[:] summation = np.zeros(N, dtype=np.complex128)
    
    cdef double p_val, f_val, d_val, angle
    
    # Multi-threaded parallel loop without GIL
    for i in prange(N - 1, nogil=True):
        p_val = p_n[i]
        f_val = f_k[i]
        d_val = data[i]
        
        if not isnan(d_val) and p_val > 0:
            angle = -2.0 * M_PI * p_val * f_val
            # Euler's formula: exp(i * angle) = cos(angle) + i * sin(angle)
            summation[i] = d_val * (cos(angle) + 1j * sin(angle))
            
    return np.asarray(summation)
