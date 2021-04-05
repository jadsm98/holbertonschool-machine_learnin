#!/usr/bin/env python3
"""module"""


import numpy as np


def markov_chain(P, s, t=1):
    """function"""

    if not isinstance(P, np.ndarray) or len(P.shape) != 2 \
       or P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.shape[1] != P.shape[0]:
        return None
    if not type(t) is int or t <= 0:
        return None
    P = np.linalg.matrix_power(P, t)
    return np.matmul(s, P)
