#!/usr/bin/env python3
"""
module
"""


def np_slice(matrix, axes={}):
    """
    function
    """

    args = []
    list_axes = []
    max_val = max(axes.keys())
    for j in range(max_val + 1):
        list_axes.append(j)
    for i in list_axes:
        if i in axes.keys():
            sl = slice(*axes[i])
        else:
            sl = slice(None)
            args.append(sl)
    return matrix[tuple(args)]
