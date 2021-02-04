#!/usr/bin/env python3
"""module"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same",
                 stride=(1, 1)):
    """function"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = W.shape[0], W.shape[1]
    c_new = W.shape[3]
    sh, sw = stride
    if padding == 'valid':
        h_new = int(np.ceil((h_prev - kh + 1)/sh))
        w_new = int(np.ceil((w_prev - kw + 1)/sw))
        A_padded = A_prev[:, :, :, :]
    elif padding == 'same':
        ph = int(np.ceil(((h_prev - 1)*sh + kh - h_prev)/2))
        pw = int(np.ceil(((w_prev - 1)*sw + kw - w_prev)/2))
        h_new = int(np.ceil(((h_prev + 2*ph - kh + 1)/sh)))
        w_new = int(np.ceil(((w_prev + 2*pw - kw + 1)/sw)))
        A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    output = np.zeros((m, h_new, w_new, c_new))
    for k in range(c_new):
        for i in range(h_new):
            for j in range(w_new):
                sliced = A_padded[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
                output[:, i, j, k] = np.sum(sliced*W[:, :, :, k],
                                            axis=(1, 2, 3))
    output = activation(output + b)
    return output
