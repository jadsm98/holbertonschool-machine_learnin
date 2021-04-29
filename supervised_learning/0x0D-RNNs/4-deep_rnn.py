#!/usr/bin/env python3
"""module"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """deep RNN"""
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    Y = []
    H[0, ...] = h_0
    for n in range(t):
        h_prev = X[n, ...]
        for layer in range(l):
            H[n + 1, layer, :, :], y = rnn_cells[layer].forward(H[n, layer, :, :], h_prev)
            h_prev = H[n + 1, layer, :, :]
        Y.append(y)
    return H, np.asarray(Y)
