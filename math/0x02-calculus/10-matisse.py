#!/usr/bin/env python3
"""
module
"""


def poly_derivative(poly):
    """
    function
    """

    if type(poly) is not list or len(poly) == 0 or poly is None:
        return None
    if not all(type(i) in [int, float] for i in poly):
        return None

    if len(poly) == 1:
        return [0]

    deriv = []
    for i in range(1, len(poly)):
        deriv.append(i*poly[i])
    return deriv
