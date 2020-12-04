#!/usr/bin/env python3
"""
module
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    function
    """
    mat3 = []

    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for i in mat1:
            mat3.append(i[:])
        for j in mat2:
            mat3.append(j[:])
        return mat3
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            mat3.append(mat1[i] + mat2[i])
        return mat3
    return None
