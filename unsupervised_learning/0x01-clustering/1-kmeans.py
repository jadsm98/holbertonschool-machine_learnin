#!/usr/bin/env python3
"""module"""


import nunpy as np


def kmeans(X, k, iterations=1000):
    """function"""
    
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not type(k) is int or k <= 0:
        return None, None
    if not type(iterations) is int or iterations <= 0:
        return None, None
    low = np.amin(X, axis=0)
    high = np.amax(X, axis=0)
    cluster = np.random.uniform(low, high, size=(k, X.shape[1]))
    for _ in range(iterations):
        clss = np.argmin(np.linalg.norm(X[:, None] - cluster, axis=-1), axis=-1)
        copy = np.copy(cluster)
        for c in range(k):
            if c not in clss:
                copy[c] = np.random.uniform(low, high)
            else:
                copy[c] = np.mean(X[clss == c], axis=0)
        if np.all(copy == cluster):
            return (cluster, clss)
        else:
            cluster = copy
    clss = np.argmin(np.linalg.norm(X[:, None] - cluster, axis=-1), axis=-1)
    return (cluster, clss)
