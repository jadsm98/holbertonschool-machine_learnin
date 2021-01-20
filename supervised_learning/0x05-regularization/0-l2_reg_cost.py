#!/usr/bin/env python3
"""module"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function"""
    W = 0
    for i in range(L):
        weight = weights['W{}'.format(i+1)]
        W += np.linalg.norm(weight)
    reg_cost = cost + (lambtha/(2*m))*W
    return reg_cost
