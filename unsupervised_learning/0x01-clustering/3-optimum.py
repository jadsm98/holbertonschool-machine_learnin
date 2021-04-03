#!/usr/bin/env python3
"""module"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """function""" 

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not type(iterations) is int or iterations < 1:
        return None, None
    if kmax is not None and (type(kmax) is not int or kmax < 1):
        return None, None
    if kmax is not None and kmin >= kmax:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if not type(kmin) is int or kmin < 1:
        return None, None
    try:
        results = []
        d_vars = []
        for k in range(kmin, kmax + 1):
            cluster, clss = kmeans(X, k, iterations)
            results.append((cluster, clss))
            variance_d = variance(X, cluster)
            if k == kmin:
                variance_k = variance_d
            d_vars.append(variance_k - variance_d)
        return results, d_vars
    except Exception:
        return None, None
