#!/usr/bin/env python3
"""module"""


import numpy as np


def sigmoid(z):
    """sigmoid function"""
    return 1/(1 + np.exp(-z))


class Neuron:
    """class Neuron"""

    def __init__(self, nx):
        """Constructor"""

        if not type(nx) is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """method"""
        z = np.matmul(self.__W, X) + self.__b
        self.__A = sigmoid(z)
        return self.__A

    def cost(self, Y, A):
        """method"""
        row, m = Y.shape
        return np.sum(-(Y*np.log(A) + (1-Y)*np.log(1.0000001 - A)))/m
