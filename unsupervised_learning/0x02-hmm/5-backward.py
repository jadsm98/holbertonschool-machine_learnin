#!/usr/bin/env python3
"""module"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """function"""
    if not isinstance(Observation, np.ndarray) or \
            len(Observation.shape) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or \
            len(Emission.shape) != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or \
            len(Transition.shape) != 2 or \
            Transition.shape[0] != Transition.shape[1]:
        return None, None
    if not isinstance(Initial, np.ndarray) or \
            len(Initial.shape) != 2 or Initial.shape[1] != 1:
        return None, None

    N, M = Emission.shape
    T = Observation.shape[0]
    b = np.zeros((N, T))
    b[:, -1] = 1
    for j in range(T-2, -1, -1):
        for i in range(N):
            b[i, j] = (b[:, j + 1] * Emission[:, Observation[j + 1]]).dot(Transition[i, :])

    like = np.sum(Initial.T[:] * Emission[:, Observation[0]] * b[:, 0])
    return like, b
