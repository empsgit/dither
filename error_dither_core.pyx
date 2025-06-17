#error_dither_core.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, language_level=3

import numpy as np
cimport numpy as np

def error_diffuse(np.ndarray[np.float32_t, ndim=3] img,
                  np.ndarray[np.int32_t, ndim=2] kernel,
                  int divisor,
                  tuple anchor,
                  np.ndarray[np.uint8_t, ndim=2] palette):

    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int kh = kernel.shape[0]
    cdef int kw = kernel.shape[1]
    cdef int ax = anchor[0]
    cdef int ay = anchor[1]
    cdef int x, y, dx, dy, nx, ny, i

    cdef float r_err, g_err, b_err
    cdef np.ndarray[np.float32_t, ndim=3] output = img.copy()

    cdef np.ndarray[np.uint8_t, ndim=1] best
    cdef float dist, min_dist
    cdef int best_idx

    for y in range(h):
        for x in range(w):
            best_idx = 0
            min_dist = 1e10
            for i in range(palette.shape[0]):
                dist = 0
                dist += (output[y, x, 0] - palette[i, 0])**2
                dist += (output[y, x, 1] - palette[i, 1])**2
                dist += (output[y, x, 2] - palette[i, 2])**2
                if dist < min_dist:
                    min_dist = dist
                    best_idx = i
            for i in range(3):
                output[y, x, i] = palette[best_idx, i]

            r_err = img[y, x, 0] - output[y, x, 0]
            g_err = img[y, x, 1] - output[y, x, 1]
            b_err = img[y, x, 2] - output[y, x, 2]

            for dy in range(kh):
                for dx in range(kw):
                    if kernel[dy, dx] == 0:
                        continue
                    ny = y + dy - ay
                    nx = x + dx - ax
                    if 0 <= nx < w and 0 <= ny < h:
                        output[ny, nx, 0] += r_err * kernel[dy, dx] / divisor
                        output[ny, nx, 1] += g_err * kernel[dy, dx] / divisor
                        output[ny, nx, 2] += b_err * kernel[dy, dx] / divisor

    return np.clip(output, 0, 255).astype(np.uint8)