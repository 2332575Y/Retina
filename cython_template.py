cimport cython
import numpy as np

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
        ret_x2 = retina_size[1]-(img_x2-input_resolution[1])
        img_x2 = input_resolution[1]
    if img_y1<0:
        ret_y1 = -img_y1
        img_y1 = 0
    if img_y2>input_resolution[0]:
        ret_y2 = retina_size[0]-(img_y2-input_resolution[0])
        img_y2 = input_resolution[0]
    return (img_x1, img_y1, img_x2, img_y2, ret_x1, ret_y1, ret_x2, ret_y2)

########################################
############## GRAY SCALE ##############
########################################

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
cpdef backProject({RESULTS}[::1] result_flat, {COEFFICIENTS}[::1] coeffs, {INDEX}[::1] idx, {BAKC_PROJECTED}[::1] back_projected):
    cdef {INDEX} x
    with nogil:
        for x in range(coeffs.shape[0]):
            if coeffs[x] > 0:
                 back_projected[x] += <{BAKC_PROJECTED}>result_flat[idx[x]]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalize({BAKC_PROJECTED}[::1] BP_flat, {NORMALIZED}[::1] norm_flat):
    cdef {INDEX} x
    cdef {NORMALIZED} c
    with nogil:
        for x in range(BP_flat.shape[0]):
            c = norm_flat[x]
            if c>1:
                BP_flat[x] = BP_flat[x]//c

#########################################
################## RGB ##################
#########################################

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
cpdef backProjectRGB({RESULTS}[::1] result_R, {RESULTS}[::1] result_G, {RESULTS}[::1] result_B, {COEFFICIENTS}[::1] coeffs, {INDEX}[::1] idx, {BAKC_PROJECTED}[::1] BP_R, {BAKC_PROJECTED}[::1] BP_G, {BAKC_PROJECTED}[::1] BP_B):
    cdef {INDEX} x, index
    cdef {COEFFICIENTS} coeff
    with nogil:
        for x in range(coeffs.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                BP_R[x] += <{BAKC_PROJECTED}>result_R[index]*coeffs[x]
                BP_G[x] += <{BAKC_PROJECTED}>result_G[index]*coeffs[x]
                BP_B[x] += <{BAKC_PROJECTED}>result_B[index]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalizeRGB({BAKC_PROJECTED}[::1] R_flat, {BAKC_PROJECTED}[::1] G_flat, {BAKC_PROJECTED}[::1] B_flat, {NORMALIZED}[::1] norm_flat):
    cdef {INDEX} x
    cdef {NORMALIZED} c
    with nogil:
        for x in range(R_flat.shape[0]):
            c = norm_flat[x]
            if c>1:
                R_flat[x] = R_flat[x]//c
                G_flat[x] = G_flat[x]//c
                B_flat[x] = B_flat[x]//c
                
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef divideRGB({RESULTS}[:,::1] result , double c):
    cdef {INDEX} x
    with nogil:
        for x in range(result.shape[1]):
            result[0,x] = <{RESULTS}>(result[0,x]//c)
            result[1,x] = <{RESULTS}>(result[1,x]//c)
            result[2,x] = <{RESULTS}>(result[2,x]//c)
    return np.asarray(result)