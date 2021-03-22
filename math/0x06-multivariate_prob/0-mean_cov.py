#!/usr/bin/env python3
"""module"""


import numpy as np


def mean_cov(X):
    """function"""
    if not isinstance(X, np.ndarray) and len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.sum(X, axis=0)/X.shape[0]
    var = X - mean
    cov = np.matmul(var.T, var)/(X.shape[0] - 1)
    return mean.reshape((1, X.shape[1])), cov
