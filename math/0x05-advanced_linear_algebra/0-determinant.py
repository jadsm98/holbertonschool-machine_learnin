#!/usr/bin/env python3
"""module"""


def determinant(matrix):
    """function"""
    total = 0

    if not type(matrix) is list or not all(type(i) is list for i in matrix):
        raise TypeError("matrix must be a list of lists")
    if not all(len(i) == len(matrix) for i in matrix) and len(matrix[0]) != 0:
        raise ValueError("matrix must be a square matrix")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) == 2 and len(matrix[0]) == 2:
        det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        return det

    for col in range(len(matrix[0])):
        M_rec = matrix.copy()
        M_rec = M_rec[1:]
        for row in range(len(M_rec)):
            M_rec[row] = M_rec[row][:col] + M_rec[row][col+1:]
        sub_det = determinant(M_rec)
        sign = (-1)**col
        total += sign * matrix[0][col] * sub_det
    return total
