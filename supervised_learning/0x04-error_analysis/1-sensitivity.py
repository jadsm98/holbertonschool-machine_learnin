#!/usr/bin/env python3
"""module"""


import numpy as np


def sensitivity(confusion):
    """function"""
    return np.divide(np.max(confusion, axis=1),
                     np.sum(confusion, axis=1))
