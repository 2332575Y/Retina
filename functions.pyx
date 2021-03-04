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
cpdef sample(unsigned char[::1] img_flat, unsigned short[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_flat):
    cdef unsigned int x
    with nogil:
        for x in range(img_flat.shape[0]):
            if coeffs[x] > 0:
                result_flat[idx[x]] += img_flat[x]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)            
cpdef backProject(unsigned int[::1] result_flat, unsigned short[::1] coeffs, unsigned int[::1] idx, unsigned long long[::1] back_projected):
    cdef unsigned int x
    with nogil:
        for x in range(coeffs.shape[0]):
            if coeffs[x] > 0:
                 back_projected[x] += <unsigned long long>result_flat[idx[x]]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalize(unsigned long long[::1] BP_flat, unsigned long long[::1] norm_flat):
    cdef unsigned int x
    cdef unsigned long long c
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
cpdef sampleRGB(unsigned char[::1] R, unsigned char[::1] G, unsigned char[::1] B, unsigned short[::1] coeffs, unsigned int[::1] idx, unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B):
    cdef unsigned int x, index
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
cpdef backProjectRGB(unsigned int[::1] result_R, unsigned int[::1] result_G, unsigned int[::1] result_B, unsigned short[::1] coeffs, unsigned int[::1] idx, unsigned long long[::1] BP_R, unsigned long long[::1] BP_G, unsigned long long[::1] BP_B):
    cdef unsigned int x, index
    cdef unsigned short coeff
    with nogil:
        for x in range(coeffs.shape[0]):
            coeff = coeffs[x]
            if coeff > 0:
                index = idx[x]
                BP_R[x] += <unsigned long long>result_R[index]*coeffs[x]
                BP_G[x] += <unsigned long long>result_G[index]*coeffs[x]
                BP_B[x] += <unsigned long long>result_B[index]*coeffs[x]

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef normalizeRGB(unsigned long long[::1] R_flat, unsigned long long[::1] G_flat, unsigned long long[::1] B_flat, unsigned long long[::1] norm_flat):
    cdef unsigned int x
    cdef unsigned long long c
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
cpdef divideRGB(unsigned int[:,::1] result , double c):
    cdef unsigned int x
    with nogil:
        for x in range(result.shape[1]):
            result[0,x] = <unsigned int>(result[0,x]//c)
            result[1,x] = <unsigned int>(result[1,x]//c)
            result[2,x] = <unsigned int>(result[2,x]//c)
    return np.asarray(result)