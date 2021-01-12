#!/usr/bin/env python3
"""module"""


import numpy as np


class DeepNeuralNetwork:
    """Deep Neural Network"""

    def __init__(self, nx, layers):
        """Constructor"""

        if not type(nx) is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not type(layers) is list:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if i == 0:
                self.weights['W{}'.format(i+1)] = np.random.randn(layers[i], nx) * np.sqrt(2/(layers[i] + nx))
                self.weights['b{}'.format(i+1)] = np.zeros([layers[i], 1])
            else:
                self.weights['W{}'.format(i+1)] = np.random.randn(layers[i], layers[i-1]) * \
                                                  np.sqrt(2/(layers[i] + layers[i-1]))
                self.weights['b{}'.format(i+1)] = np.zeros([layers[i], 1])
