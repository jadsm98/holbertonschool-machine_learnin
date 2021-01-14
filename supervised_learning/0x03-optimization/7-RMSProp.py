#!/usr/bin/env python3
"""module"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """function"""
    Sv = s * beta2 + (1 - beta2) * grad**2
    update = var - alpha * grad /(Sv**0.5 + epsilon)
    return update, Sv
