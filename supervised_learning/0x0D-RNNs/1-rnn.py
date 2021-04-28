#!/usr/bin/env python3
"""module"""

import numpy as np


def RNN(rnn_cell, X, h_0):
    """RNN forward prop"""

    o = rnn_cell.by.shape[1]
    t, m, i = X.shape
    _, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    h_prev = h_0
    H[0, :, :] = h_0
    for n in range(t):
        xt = X[n, :, :].reshape((m, i))
        h_next, y = rnn_cell.forward(h_prev, xt)
        H[n + 1, :, :] = h_next
        Y[n, :, :] = y
        h_prev = h_next
    return H, Y
