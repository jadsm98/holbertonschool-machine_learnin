#!/usr/bin/env python3
"""module"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """function"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None
    _, d = X.shape
    pi = np.full((k,), 1/k)
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))
    return pi, m, S
