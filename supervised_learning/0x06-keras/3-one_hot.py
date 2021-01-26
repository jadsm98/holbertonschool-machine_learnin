#!/usr/bin/env python3
"""module"""


import numpy as np


def one_hot(labels, classes=None):
    """function"""
    target = labels.reshape(-1)
    if classes is None:
        one_hot = np.eye(np.max(labels)+1)[target]
    else:
        one_hot = np.eye(classes)[target]
    return one_hot
