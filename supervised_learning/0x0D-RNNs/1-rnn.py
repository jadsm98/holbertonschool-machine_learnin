#!/usr/bin/env python3
"""module"""

import numpy as np


def RNN(rnn_cell, X, h_0):
    """RNN forward prop"""
    t, m, i = X.shape
    _, h = h_0.shape
    H = []
    Y = []
    h_prev = h_0
    H.append(h_0)
    for n in range(t):
        xt = X[n, :, :]
        h_prev, y = rnn_cell.forward(h_prev, xt)
        H.append(h_prev)
        Y.append(y)
    return np.asarray(H), np.asarray(Y)
