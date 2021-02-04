#!/usr/bin/env python3
"""module"""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function"""
    m, h_new, w_new, c_new = dZ.shape
    h_prev, w_prev, c_prev = A_prev.shape[1:]
    kh, kw = W.shape[0:2]
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil(((h_prev - 1)*sh + kh - h_prev)/2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev)/2))
        A_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    else:
        A_padded = A_prev[:, :, :, :]
    dA_prev = np.zeros(A_padded.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    for n in range(m):
        for k in range(c_new):
            for i in range(h_new):
                for j in range(w_new):
                    dA_prev[n, i*sh: i*sh + kh, j*sw: j*sw + kw, :] += \
                        W[:, :, :, k]*dZ[n, i, j, k]
                    A = A_prev[n, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
                    dW[:, :, :, k] += A*dZ[n, i, j, k]
                    db[:, :, :, k] += dZ[n, i, j, k]
    return dA_prev, dW, db
