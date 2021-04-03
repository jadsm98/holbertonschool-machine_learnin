#!/usr/bin/env python3
"""module"""

import numpy as np


def variance(X, C):
    """function"""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    n, d = X.shape
    k, _ = C.shape
    var = np.sum((X.reshape((1, n, d)) - C.reshape((k, 1, d)))**2,
                 axis=-1)
    minimum = np.amin(var, axis=0)
    return np.sum(minimum)
