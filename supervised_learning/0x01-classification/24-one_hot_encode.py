#!/usr/bin/env python3
"""module"""


import numpy as np


def one_hot_encode(Y, classes):
    """function"""
    if type(Y) is not np.ndarray or len(Y) == 0:
        return None
    if type(classes) is not int or classes <= np.max(Y):
        return None
    encode = np.zeros([classes, Y.shape[0]])
    for i in range(classes):
        for j in range(Y.shape[0]):
            if Y[j] == i:
                encode[i][j] = 1
    return encode
