#!/usr/bin/env python3
"""module"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """function"""
    Vd = beta1 * v + (1 - beta1) * grad
    Sv = beta2 * s + (1 - beta2) * grad ** 2
    Vd_new = Vd / (1 - beta1 ** t)
    Sv_new = Sv / (1 - beta2 ** t)
    update = var - alpha * Vd_new / (Sv_new ** 0.5 + epsilon)
    return update, Vd, Sv
