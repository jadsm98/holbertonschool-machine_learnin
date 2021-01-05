#!/usr/bin/env python3
"""module"""


import numpy as np


class NeuralNetwork:
    """class"""

    def __init__(self, nx, nodes):
        """constructor"""

        if not type(nx) is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not type(nodes) is int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1:
        """getter"""
        return self.__W1

    @property
    def b1:
        """getter"""
        return self.__b1

    @property
    def A1:
        """getter"""
        return self.__A1

    @property
    def W2:
        """getter"""
        return self.__W2

    @property
    def b2:
        """getter"""
        return self.__b2

    @property
    def A2:
        """getter"""
        return self.__A2
