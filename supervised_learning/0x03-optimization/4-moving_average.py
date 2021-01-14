#!/usr/bin/env python3
"""module"""


import numpy as np


def moving_average(data, beta):
    """function"""
    vt = []
    b = 0
    for i in range(len(data)):
        b = beta*b + (1 - beta)*data[i]
        vt.append(b / (1 - beta**(i + 1)))
    return vt
