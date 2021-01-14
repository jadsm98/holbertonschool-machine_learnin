#!/usr/bin/env python3
"""module"""


import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """function"""
    Vd = v * beta1 + (1 - beta1) * grad
    update = var - alpha * Vd
    return update, Vd
