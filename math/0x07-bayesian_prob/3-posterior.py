#!/usr/bin/env python3
"""module"""

import numpy as np


def posterior(x, n, P, Pr):
    """function"""

    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")
    n_fac = np.math.factorial(n)
    x_fac = np.math.factorial(x)
    n_x_fac = np.math.factorial(n - x)
    comb = n_fac/(x_fac*n_x_fac)
    likelihood = comb * np.power(P, x) * np.power(1 - P, n - x)
    marg = np.sum(likelihood * Pr)
    post = (likelihood * Pr)/marg
    return post
