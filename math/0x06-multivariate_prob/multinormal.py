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

    def pdf(self, x):
        """method"""
        d = x.shape[0]
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape[1] != 1\
                or x.shape[0] != self.cov.shape[0]:
            raise ValueError("x must have the shape ({}, 1)".
                             format(self.cov.shape[0]))
        inv = np.linalg.inv(self.cov)
        det = np.linalg.det(self.cov)
        denum = np.sqrt(np.power((2 * np.pi), self.cov.shape[0]) * det)
        y = np.matmul((x - self.mean).T, inv)
        pdf = (1 / denum) * np.exp(-1 * np.matmul(y, (x - self.mean)) / 2)
        return pdf.reshape(-1)[0]
