#!/usr/bin/env python3
"""module"""


import numpy as np


def sigmoid(Z):
    """function"""
    return 1/(1 + np.exp(-Z))


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
    def W1(self):
        """getter"""
        return self.__W1

    @property
    def b1(self):
        """getter"""
        return self.__b1

    @property
    def A1(self):
        """getter"""
        return self.__A1

    @property
    def W2(self):
        """getter"""
        return self.__W2

    @property
    def b2(self):
        """getter"""
        return self.__b2

    @property
    def A2(self):
        """getter"""
        return self.__A2

    def forward_prop(self, X):
        """method"""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = sigmoid(Z1)
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = sigmoid(Z2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """cost"""
        m = Y.shape[1]
        return np.sum(-(Y*np.log(A) + (1-Y)*np.log(1.0000001 - A)))/m
