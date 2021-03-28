#!/usr/bin/env python3
"""module"""


import numpy as np


def likelihood(x, n, P):
    """function"""

    if n <= 0 or not type(n) is int:
        raise ValueError("n must be a positive integer")
    if not type(x) is int or x < 0:
        raise ValueError("x must be an integer that is\
                          greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not all(P[(P >= 0) & (P <= 1)] == P):
        raise ValueError("All values in P must be in the range [0, 1]")
    n_fac = np.math.factorial(n)
    x_fac = np.math.factorial(x)
    n_x_fac = np.math.factorial(n - x)
    comb = n_fac/(x_fac*n_x_fac)
    likelihood = comb * np.power(P, x) * np.power(1 - P, n - x)
    return likelihood
