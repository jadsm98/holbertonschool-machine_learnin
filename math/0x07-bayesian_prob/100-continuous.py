#!/usr/bin/env python3
"""module"""

from scipy import special


def posterior(x, n, p1, p2):
    """function"""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if np.any(p1 < 0) or np.any(p1 > 1) or not type(p1) is float:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if np.any(p2 < 0) or np.any(p2 > 1) or not type(p2) is float:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")
    func1 = x + 1
    func2 = (n - x) + 1
    int2 = special.btdtr(func1, func2, p2)
    int1 = special.btdtr(func1, func2, p1)
    integral = int2 - int1
    return integral
