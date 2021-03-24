#!/usr/bin/env python3
"""module"""

import numpy as np


def pca(X, var=0.95):
    """function"""
    U, sig, V = np.linalg.svd(X)
    cum_sum = np.cumsum(sig)
    for i, elem in enumerate(cum_sum):
        if elem/cum_sum[-1] >= var:
            r = i
            break
    W = V.T[:, :r+1]
    return W
