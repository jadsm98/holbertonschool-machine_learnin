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
        return np.sum(-(Y*np.log(A) +
                        (1 - Y)*np.log(1.0000001 - A)))/m

    def evaluate(self, X, Y):
        """method"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        A1 = sigmoid(z1)
        z2 = np.matmul(self.__W2, A1) + self.__b2
        A2 = sigmoid(z2)
        cost = self.cost(Y, A2)
        A2 = np.where(A2 >= 0.5, 1, 0)
        return A2, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """method"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = (np.matmul(dz2, A1.T))/m
        db2 = (np.sum(dz2, axis=1, keepdims=True))/m
        dz1 = np.multiply(np.matmul(self.__W2.T, dz2), A1)
        dw1 = (np.matmul(dz1, X.T))/m
        db1 = (np.sum(dz1, axis=1, keepdims=True))/m
        self.__W1 -= alpha*dw1
        self.__W2 -= alpha*dw2
        self.__b1 -= alpha*db1
        self.__b2 -= alpha*db2

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """method"""
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
            z1 = np.matmul(self.__W1, X) + self.__b1
            self.__A1 = sigmoid(z1)
            z2 = np.matmul(self.__W2, self.__A1) + self.__b2
            self.__A2 = sigmoid(z2)
            cost = self.cost(Y, self.__A2)
            dz2 = self.__A2 - Y
            dw2 = (np.matmul(dz2, self.__A1.T))/m
            db2 = (np.sum(dz2, axis=1, keepdims=True))/m
            dz1 = np.multiply(np.matmul(self.__W2.T, dz2), self.__A1)
            dw1 = (np.matmul(dz1, X.T))/m
            db1 = (np.sum(dz1, axis=1, keepdims=True))/m
            self.__A2 = np.where(self.__A2 >= 0.5, 1, 0)
            self.__W1 -= alpha*dw1
            self.__W2 -= alpha*dw2
            self.__b1 -= alpha*db1
            self.__b2 -= alpha*db2
        return self.__A2, cost
