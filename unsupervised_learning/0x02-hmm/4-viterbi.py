#!/usr/bin/env python3
"""module"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
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
    backstart = np.zeros(T)
    v = np.zeros((N, T))
    backpoint = np.zeros((N, T))
    v[:, 0] = np.multiply(Initial.reshape(-1), Emission[:, Observation[0]])
    for j in range(1, T):
        for i in range(N):
            v[i, j] = np.amax(v[:, j - 1] * Transition[:, i]) * \
                      Emission[i, Observation[j]]
            backpoint[i, j] = np.argmax(v[:, j - 1] * Transition[:, i])

    P = np.amax(v[:, -1])
    backstart[-1] = np.argmax(v[:, -1])
    for i in range(T - 2, -1, -1):
        backstart[i] = backpoint[int(backstart[i + 1]), i + 1]
    path = [int(i) for i in backstart]
    return path, P
