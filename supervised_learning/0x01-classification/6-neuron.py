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
        m = Y.shape[1]
        return np.sum(-(Y*np.log(A) + (1-Y)*np.log(1.0000001 - A)))/m

    def evaluate(self, X, Y):
        """method"""
        self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        self.__A = np.where(self.__A >= 0.5, 1, 0)
        return self.__A, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """gradient descent"""
        m = Y.shape[1]
        dz = A[:] - Y
        dw = (np.matmul(dz, X.T))/m
        db = (np.sum(dz))/m
        self.__W -= alpha*dw
        self.__b -= alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """trains the neuron"""
        if not type(iterations) is int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not type(alpha) is float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        m = Y.shape[1]
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha=0.05)
        return self.evaluate(X, Y)
