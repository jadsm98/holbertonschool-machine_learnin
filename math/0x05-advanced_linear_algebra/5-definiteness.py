#!/usr/bin/env python3


import numpy as np


def definiteness(matrix):
    """function"""
    if not isinstance(marix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if not any(len(i) == len(matrix) for i in matrix) or len(matrix) == 0:
        return None
    if not any(len(i) == len(matrix) for i in matrix) or len(matrix) == 0:
        return None
    eigen = np.linalg.eig(matrix)[0]
    if all(i > 0 for i in eigen):
        return "Positive definite"
    if all(i >= 0 for i in eigen):
        return "Positive semi-definite"
    if all(i < 0 for i in eigen):
        return "Negative definite"
    if all(i <= 0 for i in eigen):
        return "Negative semi-definite"
    return "Indefinite"
