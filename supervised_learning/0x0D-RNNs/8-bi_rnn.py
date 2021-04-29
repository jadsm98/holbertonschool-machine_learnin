#!/usr/bin/env python3
"""module"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """bidirectional RNN"""
    t, m, i = X.shape
    _, h = h_0.shape
    H1 = []
    h_prev = h_0
    for n in range(t):
        xt = X[n, :, :]
        h_prev = bi_cell.forward(h_prev, xt)
        H1.append(h_prev)
    h_next = h_t
    H2 = []
    for n in range(t - 1, -1, -1):
        xt = X[n, :, :]
        h_next = bi_cell.backward(h_next, xt)
        H2.append(h_next)
    H2.reverse()
    H1 = np.asarray(H1)
    H2 = np.asarray(H2)
    H = np.concatenate((H1, H2), axis=-1)
    Y = bi_cell.output(np.asarray(H))
    return H, Y
