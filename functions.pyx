cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef zeros_int32(unsigned int[::1] arr):
    cdef unsigned int x
    for x in range(arr.shape[0]):
        arr[x] = 0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalize(unsigned int[::1] BP_flat, unsigned int[::1] norm_flat):
    cdef unsigned int x
    for x in range(BP_flat.shape[0]):
        BP_flat[x] = BP_flat[x]//norm_flat[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef sample(unsigned char[::1] img_flat, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_flat):
    cdef unsigned int x
    for x in range(img_flat.shape[0]):
        if coeffs[x] > 0:
            result_flat[idx[x]] += img_flat[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef sampleRGB(unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B):
    cdef unsigned int x, index
    cdef unsigned char coeff
    for x in range(R.shape[0]):
        coeff = coeffs[x]
        if coeff > 0:
            index = idx[x]
            result_R[index] += R[x]*coeff
            result_G[index] += G[x]*coeff
            result_B[index] += B[x]*coeff

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef backProject(unsigned int[::1] result_flat, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] back_projected):
    cdef unsigned int x
    for x in range(back_projected.shape[0]):
        if coeffs[x] > 0:
             back_projected[x] += result_flat[idx[x]]*coeffs[x]
                    
@cython.wraparound(False)
@cython.boundscheck(False)
cpdef backProjectRGB(unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B, unsigned char[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] BP_R, unsigned int[::1] BP_G, unsigned int[::1] BP_B):
    cdef unsigned int x, index
    cdef unsigned char coeff
    for x in range(BP_R.shape[0]):
        coeff = coeffs[x]
        if coeff > 0:
            index = idx[x]
            BP_R[x] += result_R[index]*coeffs[x]
            BP_G[x] += result_G[index]*coeffs[x]
            BP_B[x] += result_B[index]*coeffs[x]