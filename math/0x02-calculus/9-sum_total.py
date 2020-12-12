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

    return (n*(2*n + 1)*(n + 1))//6
