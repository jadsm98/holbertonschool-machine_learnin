#!/usr/bin/env python3
"""module"""


import numpy as np


def one_hot(labels, classes=None):
    """function"""
    if classes is None:
        maximum = np.max(labels)
        one_hot = np.zeros((labels.shape[0], maximum + 1))
        one_hot[np.arange(labels.shape[0]), labels] = 1
    else:
        one_hot = np.zeros((labels.shape[0], classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot
