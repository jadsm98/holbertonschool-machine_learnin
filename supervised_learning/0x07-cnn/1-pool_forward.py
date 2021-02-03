#!/usr/bin/env python3
"""module"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_new = int(np.ceil((h_prev - kh + 1) / sh))
    w_new = int(np.ceil((w_prev - kw + 1) / sw))
    output = np.zeros((m, h_new, w_new, c_prev))
    for i in range(h_new):
        for j in range(w_new):
            sliced = A_prev[:, i * sh: i * sh + kh, j * sw: j * sw + kw, :]
            if mode == 'max':
                output[:, i, j, :] = np.max(sliced, axis=(1, 2))
            else:
                output[:, i, j, :] = np.mean(sliced, axis=(1, 2))
    return output
