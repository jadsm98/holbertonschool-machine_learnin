#!/usr/bin/env python3
"""module"""


import numpy as np


def P_init(X, perplexity):
    """function"""
    n, d = X.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.power(np.linalg.norm(X[i,:] - X[j,:]), 2)
    P = np.zeros((n, n))
    betas = np.ones(n).reshape((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)
