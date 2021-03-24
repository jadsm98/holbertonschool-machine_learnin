#!/usr/bin/env python3
"""module"""


import numpy as np


def HP(Di, beta):
    """function"""
    num = np.exp(np.divide(-Di, beta))
    denum = np.sum(num)
    Pi = num/denum
    Hi = - np.sum(Pi * np.log2(Pi))
    return Hi, Pi
