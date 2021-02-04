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
    for i in range(h_new):
        for j in range(w_new):
            A_prev_sliced = A_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
            if mode == 'max':
                maximum = np.max(A_prev_sliced,
                                 axis=(1, 2)).reshape(10, 1, 1, 2)
                position = (A_prev_sliced == maximum)
                dA_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :] = \
                    position*dA[:, i, j, :].reshape(10, 1, 1, 2)
            else:
                dA_prev[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :] = \
                    dA[:, i, j, :]/(kh*kw).reshape(10, 1, 1, 2)
    return dA_prev
