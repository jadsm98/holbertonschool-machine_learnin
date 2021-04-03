#!/usr/bin/env python3
"""module"""

import numpy as np


def initialize(X, k):
    """function"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not type(k) is int or k <= 0:
        return None
    try:
        min = np.amin(X, axis=0)
        max = np.amax(X, axis=0)
        clusters = np.random.uniform(low=min, high=max,
                                     size=(k, X.shape[1]))
        return clusters
    except Exception:
        return None
