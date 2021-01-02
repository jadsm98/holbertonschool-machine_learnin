#!/usr/bin/env python3
"""
module
"""


import numpy as np


class Neuron:
    """
    class Neuron
    """

    def __init__(self, nx):
        """
        Constructor
        """

        if not type(nx) is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
