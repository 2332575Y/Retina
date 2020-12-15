cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef zeros_int32({RESULTS}[::1] arr):
    cdef {INDEX} x
    for x in range(arr.shape[0]):
        arr[x] = 0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalize({BAKC_PROJECTED}[::1] BP_flat, {NORMALIZED}[::1] norm_flat):
    cdef {INDEX} x
    for x in range(BP_flat.shape[0]):
        BP_flat[x] = BP_flat[x]//norm_flat[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef sample({INPUT}[::1] img_flat, {COEFFICIENTS}[::1] coeffs, {INDEX}[::1] idx, {RESULTS}[::1] result_flat):
    cdef {INDEX} x
    with nogil:
        for x in range(img_flat.shape[0]):
            if coeffs[x] > 0:
                result_flat[idx[x]] += img_flat[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef sampleRGB({INPUT}[::1] R, {INPUT}[::1] G, {INPUT}[::1] B, {COEFFICIENTS}[::1] coeffs, {INDEX}[::1] idx, {RESULTS}[::1] result_R, {RESULTS}[::1] result_G, {RESULTS}[::1] result_B):
    cdef {INDEX} x, index
    cdef {COEFFICIENTS} coeff
    with nogil:
        for x in range(R.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                result_R[index] += R[x]*coeff
                result_G[index] += G[x]*coeff
                result_B[index] += B[x]*coeff

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef backProject({RESULTS}[::1] result_flat, {COEFFICIENTS}[::1] coeffs, {INDEX}[::1] idx, {BAKC_PROJECTED}[::1] back_projected):
    cdef {INDEX} x
    with nogil:
        for x in range(back_projected.shape[0]):
            if coeffs[x] > 0:
                 back_projected[x] += result_flat[idx[x]]*coeffs[x]
                    
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef backProjectRGB({RESULTS}[::1] result_R, {RESULTS}[::1] result_G, {RESULTS}[::1] result_B, {COEFFICIENTS}[::1] coeffs, {INDEX}[::1] idx, {BAKC_PROJECTED}[::1] BP_R, {BAKC_PROJECTED}[::1] BP_G, {BAKC_PROJECTED}[::1] BP_B):
    cdef {INDEX} x, index
    cdef {COEFFICIENTS} coeff
    with nogil:
        for x in range(BP_R.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                BP_R[x] += result_R[index]*coeffs[x]
                BP_G[x] += result_G[index]*coeffs[x]
                BP_B[x] += result_B[index]*coeffs[x]