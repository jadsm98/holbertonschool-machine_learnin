#!/usr/bin/env python3
"""
module
"""


def add_arrays(arr1, arr2):
    """
    function
    """

    arr3 = []

    if len(arr1) != len(arr2):
        return None

    for i in range(len(arr1)):
        arr3.append(arr1[i] + arr2[i])
    return arr3
