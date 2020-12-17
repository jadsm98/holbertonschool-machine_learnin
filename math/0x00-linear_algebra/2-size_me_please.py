#!/usr/bin/env python3
"""
module
"""


def matrix_shape(matrix):
    """
    function
    """
    shape = []
    shape.append(len(matrix))
    i = matrix[0]
    while True:
        if type(i) is list:
            shape.append(len(i))
        else:
            break
        i = i[0]
    return shape
