#!/usr/bin/env python3
"""module"""


import numpy as np


def one_hot_decode(one_hot):
    """function"""
    if type(one_hot) is not np.ndarray or len(one_hot) == 0:
        return None
    if len(one_hot.shape) != 2:
        return None
    trans = list(one_hot.T)
    decode = []
    for i in range(len(trans)):
        for j in range(len(trans[0])):
            if trans[i][j] == 1:
                decode.append(j)
    return np.asarray(decode)
