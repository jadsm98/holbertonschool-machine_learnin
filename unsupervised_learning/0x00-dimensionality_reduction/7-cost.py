#!/usr/bin/env python3
"""module"""

import numpy as np


def cost(P, Q):
    """function"""
    C = np.sum(P*np.log(P/Q), axis=(0,1))
    return C
