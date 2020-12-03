#!/usr/bin/env python3
"""
module
"""


def np_cat(mat1, mat2, axis=0):
    """
    function
    """

    import numpy as np

    if axis == 0:
        return np.vstack((mat1, mat2))
    return np.hstack((mat1, mat2))
