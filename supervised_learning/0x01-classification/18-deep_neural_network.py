#!/usr/bin/env python3
"""module"""


import numpy as np


def sigmoid(z):
    """sigmoid function"""
    return 1/(1 + np.exp(-z))



class DeepNeuralNetwork:
    """Deep Neural Network"""

    def __init__(self, nx, layers):
        """Constructor"""

        if not type(nx) is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not type(layers) is list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if layers[i] <= 0 or not type(layers[i]) is int:
                raise TypeError("layers must be a list of positive integers")
            if i == 0:
                self.__weights['W{}'.format(i+1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2/(nx))
                self.__weights['b{}'.format(i+1)] = np.zeros([layers[i], 1])
            else:
                self.__weights['W{}'.format(i+1)] = \
                    np.random.randn(layers[i], layers[i-1]) * \
                    np.sqrt(2/(layers[i-1]))
                self.__weights['b{}'.format(i+1)] = np.zeros([layers[i], 1])

    @property
    def L(self):
        """getter"""
        return self.__L

    @property
    def cache(self):
        """getter"""
        return self.__cache

    @property
    def weights(self):
        """getter"""
        return self.__weights

    def forward_prop(self, X):
        """forward propagation"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            z = np.matmul(self.__weights['W{}'.format(i+1)],
                          self.__cache['A{}'.format(i)]) + \
                self.__weights['b{}'.format(i+1)]
            self.__cache['A{}'.format(i+1)] = sigmoid(z)
        return self.__cache['A{}'.format(self.__L)], self.__cache
