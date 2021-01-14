#!/usr/bin/env python3
"""module"""


import numpy as np


def shuffle_data(X, Y):
    """function"""
    X_shuffled = np.random.permutation(X)
    Y_shuffled = np.random.permutation(Y)
    return X_shuffled, Y_shuffled
