#!/usr/bin/env python3
"""module"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_new, w_new, c_new = dA.shape[1:]
    dA_prev = np.zeros(A_prev.shape)
    for n in range(m):
        for k in range(c_new):
            for i in range(h_new):
                for j in range(w_new):
                    if mode == 'max':
                        A_prev_sliced = A_prev[n, i*sh: i*sh + kh,
                                               j*sw: j*sw + kw, k]
                        position = (A_prev_sliced == np.max(A_prev_sliced))
                        dA_prev[n, i*sh: i*sh + kh, j*sw: j*sw + kw, k] += \
                            position*dA[n, i, j, k]
                    else:
                        dA_prev[n, i*sh: i*sh + kh, j*sw: j*sw + kw, k] += \
                            dA[n, i, j, k]/(kh*kw)
    return dA_prev
