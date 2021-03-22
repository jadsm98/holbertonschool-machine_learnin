#!/usr/bin/env python3
"""module"""


import numpy as np


class MultiNormal:
    """class"""

    def __init__(self, data):
        """initializer"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape((data.shape[0], 1))
        data_t = data.T
        mean = np.mean(data_t, axis=0)
        var = data_t - mean
        self.cov = np.matmul(var.T, var)/(data_t.shape[0] - 1)
