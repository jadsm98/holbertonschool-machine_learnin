#!/usr/bin/env python3
"""module"""


import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """function"""
    cache = {'A0': X}
    for i in range(L):
        z = np.matmul(weights['W{}'.format(i + 1)],
                      cache['A{}'.format(i)]) \
            + weights['b{}'.format(i + 1)]
        if i == L - 1:
            a = np.exp(z)/(np.sum(np.exp(z), axis=0, keepdims=True))
        else:
            a = 2/(1 + np.exp(-2*z)) - 1
            d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
            a *= d
            a /= keep_prob
            cache['D{}'.format(i + 1)] = d.astype(int)
        cache['A{}'.format(i + 1)] = a
    return cache
