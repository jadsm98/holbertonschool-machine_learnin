#!/usr/bin/env python3
"""module"""


import numpy as np


def normalize(X, m, s):
    """function"""
    return np.divide(X - m, s)
