#!/usr/bin/env python3
"""
module
"""


def add_matrices2D(mat1, mat2):
    """
    function
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    
    mat3 = [[0 for i in mat1[0]] for j in mat1]

    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            mat3[i][j] = mat1[i][j] + mat2[i][j]
    return mat3
