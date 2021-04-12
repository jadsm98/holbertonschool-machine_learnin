#!/usr/bin/env python3
"""module"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """function"""
    alpha = np.zeros((Emission.shape[0], Observation.shape[0]))
    for s in range(Initial.shape[0]):
        alpha[s, 0] = Initial[s] * Emission[s, Observation[0]]
    for t in range(1, Observation.shape[0]):
        for j in range(Emission.shape[0]):
            alpha[j, t] = alpha[:, t - 1].dot(Transition[:, j]) * \
                            Emission[j, Observation[t]]
    likelihood = np.sum(alpha[:, -1])
    return likelihood, alpha


def backward(Observation, Emission, Transition, Initial):
    """function"""
    N, M = Emission.shape
    T = Observation.shape[0]
    b = np.zeros((N, T))
    b[:, -1] = 1
    for j in range(T-2, -1, -1):
        for i in range(N):
            b[i, j] = (b[:, j + 1] *
                       Emission[:, Observation[j + 1]]).dot(Transition[i, :])
    like = np.sum(Initial.T[:] * Emission[:, Observation[0]] * b[:, 0])
    return like, b


def baum_welch(Observations, Transition, Emission, Initial,
               iterations=1000):
    """function"""
    if not isinstance(Observations, np.ndarray) or \
            len(Observations.shape) != 1:
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
    T = Observations.shape[0]

    for n in range(iterations):
        _, alpha = forward(Observations, Emission, Transition, Initial)
        _, beta = backward(Observations, Emission, Transition, Initial)
        etta = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            a = np.matmul(alpha[:, t].T, Transition)
            b = Emission[:, Observations[t + 1]].T
            c = beta[:, t + 1]
            denom = np.matmul(a * b, c)
            for i in range(N):
                num = alpha[i, t] * Transition[i] * \
                            Emission[:, Observations[t + 1]].T *\
                            beta[:, t + 1].T
                etta[i, :, t] = num / denom
        gamma = np.sum(etta, axis=1)
        Transition = np.sum(etta, 2)/np.sum(gamma, axis=1).reshape((-1, 1))
        sum_etta = np.sum(etta[:, :, T - 2], axis=0).reshape((-1, 1))
        gamma = np.hstack((gamma, sum_etta))
        denominator = np.sum(gamma, axis=1).reshape((-1, 1))
        for i in range(M):
            gamma_i = gamma[:, Observations == i]
            Emission[:, i] = np.sum(gamma_i, axis=1)
        Emission = Emission / denominator

    return Transition, Emission
