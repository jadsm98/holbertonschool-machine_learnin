#!/usr/bin/env python3
"""module"""


import numpy as np


def specificity(confusion):
    """function"""
    diag = confusion.diagonal()
    FN = np.sum(confusion, axis=1)
    FP = np.add(np.sum(confusion, axis=0), - diag)
    TN = np.sum(confusion) - FN - FP
    return np.divide(TN, np.add(TN, FP))
