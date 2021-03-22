#!/usr/bin/env python3
"""module"""


import numpy as np


def correlation(C):
    """function"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    diag = np.sqrt(np.diag(C))
    corr = np.zeros(C.shape)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            corr[i][j] = C[i][j]/(diag[i]*diag[j])
    return corr
