#!/usr/bin/env python3
"""module"""


import numpy as np


def dropout_gradient_descent(Y, weights, cache,
                             alpha, keep_prob, L):
    """function"""
    copied = weights.copy()
    m = Y.shape[1]
    for i in range(L, 0, -1):
        if i == L:
            dz = cache['A{}'.format(L)] - Y
        else:
            dz = np.multiply(np.matmul(
                copied['W{}'.format(i + 1)].T, dz),
                - np.square(cache['A{}'.format(i)]) + 1)
            dz *= cache['D{}'.format(i)]
            dz /= keep_prob
        dw = np.matmul(dz, cache['A{}'.format(i - 1)].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        weights['W{}'.format(i)] = weights['W{}'.format(i)] \
            - alpha * dw
        weights['b{}'.format(i)] = weights['b{}'.format(i)] \
            - alpha * db
