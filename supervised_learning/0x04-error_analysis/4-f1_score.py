#!/usr/bin/env python3
"""module"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """function"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    mul = np.multiply(sens, prec)
    add = np.add(sens, prec)
    return 2 * np.divide(mul, add)
