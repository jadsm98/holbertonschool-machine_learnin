#!/usr/bin/env python3
"""
module
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    function
    """

    if axis == 0:
        return np.vstack((mat1, mat2))
    return np.hstack((mat1, mat2))
