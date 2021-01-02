#!/usr/bin/env python3
"""                                                                                                                    module                                                                                                                """


import numpy as np


def sigmoid(z):
    """sigmoid function"""
    return 1/(1 + np.exp(-z))
    

class Neuron:
    """                                                                                                                    class Neuron                                                                                                           """

    def __init__(self, nx):
        """                                                                                                                    Constructor                                                                                                            """

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
        m = Y.shape[1]
        return np.sum(-(Y*np.log(A) + (1-Y)*np.log(1.0000001 - A)))/m

    def evaluate(self, X, Y):
        """method"""
        z = np.matmul(self.__W, X) + self.__b
        A = sigmoid(z)
        cost = np.sum(-(Y*np.log(A) + (1-Y)*np.log(1.0000001 - A)))/(Y.shape[1])
        A = np.where(A >= 0.5, 1, 0)
        return A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent"""
        m = Y.shape[1]
        dz = A - Y
        dw = (np.matmul(dz, X.T))/m
        db = (np.sum(dz))/m
        self.__W -= alpha*dw
        self.__b -= alpha*db
