#!/usr/bin/env python3
"""module"""


import numpy as np


def shuffle_data(X, Y):
    """function"""
    s = np.arange(X.shape[0])
    perm = np.random.permutation(s)
    return X[perm], Y[perm]
