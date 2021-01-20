#!/usr/bin/env python3
"""module"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """method"""
    m = Y.shape[1]
    for i in range(L, 0, -1):
        if i == L:
            dz = cache['A{}'.format(L)] - Y
        else:
            dz = np.multiply(np.matmul(
                weights['W{}'.format(i + 1)].T, dz),
                cache['A{}'.format(i)] * (1 - cache['A{}'.format(i)]))
        dw = np.matmul(dz, cache['A{}'.format(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W{}'.format(i)] = \
            (1 - (alpha * lambtha)/m) * weights['W{}'.format(i)] - \
            alpha * dw
        weights['b{}'.format(i)] = \
            weights['b{}'.format(i)] - alpha * db
