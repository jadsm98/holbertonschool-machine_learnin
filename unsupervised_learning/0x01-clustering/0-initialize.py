#!/usr/bin/env python3
"""module"""

import numpy as np


def initialize(X, k):
    """function"""
    try:
        min = np.amin(X, axis=0)
        max = np.amax(X, axis=0)
        clusters = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))
        return clusters
    except:
        return None
