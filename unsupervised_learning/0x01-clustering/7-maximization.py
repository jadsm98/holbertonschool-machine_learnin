#!/usr/bin/env python3
"""module"""


import numpy as np

def maximization(X, g):
    """function"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None
    n, d = X.shape
    k = g.shape[0]
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for ki in range(k):
        density = np.sum(g[ki])
        pi[ki] = density / n
        m[ki] = np.sum(np.matmul(g[ki].reshape(1, n), X), axis=0) / density
        dif = (X - m[ki])
        S[ki] = np.dot(g[ki].reshape(1, n) * dif.T, dif) / density

    return pi, m, S
