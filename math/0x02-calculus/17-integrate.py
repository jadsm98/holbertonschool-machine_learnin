#!/usr/bin/env python3
"""
module
"""


def poly_integral(poly, C=0):
    """
    function
    """


    if type(poly) is not list:
        return None
    if not all(type(i) in [int, float] for i in poly):
        return None
    if not type(C) in [int, float]:
        return None

    integ = [C]
    for i in range(len(poly)):
        coef = poly[i]/(i + 1)
        if round(coef) == coef:
            coef = int(coef)
        integ.append(coef)
    return integ
