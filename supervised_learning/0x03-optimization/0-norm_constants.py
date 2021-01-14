#!/usr/bin/env python3
"""module"""


import numpy as np


def normalization_constants(X):
    """function"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
