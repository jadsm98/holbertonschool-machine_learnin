#!/usr/bin/env python3
""" module """


def summation_i_squared(n):
    """
    function
    """

    if type(n) is not int:
        return None
    if n < 1:
        return None

    if n == 1:
        return 1
    return n**2 + summation_i_squared(n-1)
