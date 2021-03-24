#!/usr/bin/env python3
"""module"""

import numpy as np


def pca(X, ndim):
    """function"""
    mean = np.mean(X, axis=0)
    _, _, V = np.linalg.svd(mean)
    W = V.T[:, :ndim]
    return np.matmul(mean, W)
