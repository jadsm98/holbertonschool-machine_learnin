#!/usr/bin/env python3
"""module"""


import numpy as np


def Q_affinities(Y):
    """function"""
    n, ndim = Y.shape
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i][j] = np.power(np.linalg.norm(Y[i, :] - Y[j, :]), 2)
    num = 1/(1 + D)
    denum = np.sum(num)
    Q = num/denum
    return Q, num
