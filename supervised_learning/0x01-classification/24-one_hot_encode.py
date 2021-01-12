#!/usr/bin/env python3
"""module"""


import numpy as np


def one_hot_encode(Y, classes):
    """function"""
    encode = np.zeros([classes, Y.shape[0]])
    for i in range(classes):
        for j in range(Y.shape[0]):
            if Y[j] == i:
                encode[i][j] = 1
    return encode
