cimport cython
#from cython.parallel import parallel, prange

@cython.wraparound(False)
@cython.boundscheck(False)    
cpdef get_bounds(int[::1] input_resolution, int[::1]retina_size, int[::1]fixation):
    cdef int img_x1, img_y1, img_x2, img_y2, ret_x1, ret_y1, ret_x2, ret_y2
    img_x1, img_y1, img_x2, img_y2 = fixation[0]-(retina_size[1]//2), fixation[1]-(retina_size[0]//2), fixation[0]+(retina_size[1]//2), fixation[1]+(retina_size[0]//2)
    ret_x1, ret_y1, ret_x2, ret_y2 = 0, 0, retina_size[1], retina_size[0]
    if img_x1<0:
        ret_x1 = -img_x1
        img_x1 = 0
    if img_x2>input_resolution[1]:
        ret_x2 = img_x2-input_resolution[1]
        img_x2 = input_resolution[1]
    if img_y1<0:
        ret_y1 = -img_y1
        img_y1 = 0
    if img_y2>input_resolution[0]:
        ret_y2 = img_y2-input_resolution[0]
        img_y2 = input_resolution[0]
    return (img_x1, img_y1, img_x2, img_y2, ret_x1, ret_y1, ret_x2, ret_y2)

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef zeros_int32(unsigned long long[::1] arr):
    cdef unsigned long x
    with nogil:
        for x in range(arr.shape[0]):
            arr[x] = 0

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalize(unsigned long long[::1] BP_flat, unsigned long long[::1] norm_flat):
    cdef unsigned long x
    with nogil:
        for x in range(BP_flat.shape[0]):
            BP_flat[x] = BP_flat[x]//norm_flat[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef sample(unsigned char[::1] img_flat, unsigned short[::1] coeffs, unsigned long[::1] idx, unsigned long long[::1] result_flat):
    cdef unsigned long x
    with nogil:
        for x in range(img_flat.shape[0]):
            if coeffs[x] > 0:
                result_flat[idx[x]] += img_flat[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef backProject(unsigned long long[::1] result_flat, unsigned short[::1] coeffs, unsigned long[::1] idx, unsigned long long[::1] back_projected):
    cdef unsigned long x
    with nogil:
        for x in range(coeffs.shape[0]):
            if coeffs[x] > 0:
                 back_projected[x] += result_flat[idx[x]]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef sampleRGB(unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned short[::1] coeffs, unsigned long[::1] idx, unsigned long long[::1] result_R, unsigned long long[::1] result_G, unsigned long long[::1] result_B):
    cdef unsigned long x, index
    cdef unsigned short coeff
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
cpdef backProjectRGB(unsigned long long[::1] result_R, unsigned long long[::1] result_G, unsigned long long[::1] result_B, unsigned short[::1] coeffs, unsigned long[::1] idx, unsigned long long[::1] BP_R, unsigned long long[::1] BP_G, unsigned long long[::1] BP_B):
    cdef unsigned long x, index
    cdef unsigned short coeff
    with nogil:
        for x in range(BP_R.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                BP_R[x] += result_R[index]*coeffs[x]
                BP_G[x] += result_G[index]*coeffs[x]
                BP_B[x] += result_B[index]*coeffs[x]
                
####################################################
# @cython.wraparound(False)
# @cython.boundscheck(False)            
# cpdef sample_parallel(unsigned char[::1] img_flat, unsigned short[::1] coeffs, unsigned long[::1] idx, unsigned long long[::1] result_flat, unsigned int nThreads):
#     cdef unsigned long x
#     with nogil, parallel(num_threads=nThreads):
#         for x in prange(img_flat.shape[0]):
#             if coeffs[x] > 0:
#                 result_flat[idx[x]] += img_flat[x]*coeffs[x]