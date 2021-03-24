#!/usr/bin/env python3
"""module"""

import numpy as np


def pca(X, ndim):
    """function"""
    _, _, V = np.linalg.svd(X)
    W = V.T[:, :ndim+1]
    return np.matmul(X, W)
