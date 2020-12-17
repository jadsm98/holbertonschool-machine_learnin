#!/usr/bin/env python3
"""
module
"""


def matrix_transpose(matrix):
    """
    function
    """

    matrix_t = [[0 for i in matrix] for j in matrix[0]]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix_t[j][i] = matrix[i][j]
    return matrix_t
