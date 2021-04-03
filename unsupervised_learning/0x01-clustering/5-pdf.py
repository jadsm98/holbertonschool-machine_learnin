#!/usr/bin/env python3
"""module"""


import numpy as np


def pdf(X, m, S):
    """function"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != m.shape[0] or X.shape[1] != S.shape[0]:
        return None
    if S.shape[0] != S.shape[1]:
        return None
    inv = np.linalg.inv(S)
    det = np.linalg.det(S)
    denum = np.sqrt(np.power((2 * np.pi), S.shape[0]) * det)
    y = np.dot((X - m), inv)
    pdf = (1 / denum) * np.exp(np.sum((-1/2) * y * (X - m), axis=1))
    return np.where(pdf < 1e-300, 1e-300, pdf)
