#!/usr/bin/env python3
"""module"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
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
    alpha = np.zeros((Emission.shape[0], Observation.shape[0]))
    for s in range(Initial.shape[0]):
        alpha[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for t in range(1, Observation.shape[0]):
        for j in range(Emission.shape[0]):
            alpha[j, t] = alpha[:, t - 1].dot(Transition[:, j]) * \
                            Emission[j, Observation[t]]
    likelihood = np.sum(alpha[:, -1])
    return likelihood, alpha
