# dither_core.pyx
# cython: boundscheck=False, wraparound=False, nonecheck=False, initializedcheck=False, language_level=3

import numpy as np
cimport numpy as np

def atkinson_dither(np.ndarray[np.float32_t, ndim=3] img,
                    np.ndarray[np.uint8_t, ndim=2] palette_rgb):
    cdef int h = img.shape[0]
    cdef int w = img.shape[1]
    cdef int i, j, dx, dy, nx, ny, best_idx
    cdef float min_dist, d
    cdef float error[3]
    cdef np.ndarray[np.uint8_t, ndim=3] out_img = np.zeros((h, w, 3), dtype=np.uint8)

    cdef int offsets[6][2]
    offsets[0][0], offsets[0][1] = 1, 0
    offsets[1][0], offsets[1][1] = 2, 0
    offsets[2][0], offsets[2][1] = -1, 1
    offsets[3][0], offsets[3][1] = 0, 1
    offsets[4][0], offsets[4][1] = 1, 1
    offsets[5][0], offsets[5][1] = 0, 2

    cdef float r, g, b, pr, pg, pb

    for y in range(h):
        for x in range(w):
            r = img[y, x, 0]
            g = img[y, x, 1]
            b = img[y, x, 2]

            min_dist = 1e10
            best_idx = 0
            for i in range(palette_rgb.shape[0]):
                pr = palette_rgb[i, 0]
                pg = palette_rgb[i, 1]
                pb = palette_rgb[i, 2]
                d = (r - pr) ** 2 + (g - pg) ** 2 + (b - pb) ** 2
                if d < min_dist:
                    min_dist = d
                    best_idx = i

            out_img[y, x, 0] = palette_rgb[best_idx, 0]
            out_img[y, x, 1] = palette_rgb[best_idx, 1]
            out_img[y, x, 2] = palette_rgb[best_idx, 2]

            error[0] = (r - palette_rgb[best_idx, 0]) / 8.0
            error[1] = (g - palette_rgb[best_idx, 1]) / 8.0
            error[2] = (b - palette_rgb[best_idx, 2]) / 8.0

            for i in range(6):
                dx = offsets[i][0]
                dy = offsets[i][1]
                nx = x + dx
                ny = y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img[ny, nx, 0] += error[0]
                    img[ny, nx, 1] += error[1]
                    img[ny, nx, 2] += error[2]

    return out_img