#!/usr/bin/env python3
"""module"""


import numpy as np


def one_hot(labels, classes=None):
    """function"""
    if type(labels) is not np.ndarray or len(labels) == 0:
        return None
    if classes is None:
        classes = np.max(labels) + 1
    A = np.eye(classes)[labels]
    return A
