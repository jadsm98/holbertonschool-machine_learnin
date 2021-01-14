#!/usr/bin/env python3
"""module"""


import numpy as np
import matplotlib.pyplot as plt
import pickle


def sigmoid(z):
    """sigmoid function"""
    return 1/(1 + np.exp(-z))


def tanh(z):
    """tanh function"""
    return 2 * (np.exp(2*z)/(1 + np.exp(2*z))) - 1


class DeepNeuralNetwork:
    """Deep Neural Network"""

    def __init__(self, nx, layers, activation='sig'):
        """Constructor"""

        if not type(nx) is int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not type(layers) is list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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

    @property
    def activation(self):
        """getter"""
        return self.__activation


    def forward_prop(self, X):
        """forward propagation"""
        self.__cache['A0'] = X
        for i in range(self.__L):
            z = np.matmul(self.__weights['W{}'.format(i+1)],
                          self.__cache['A{}'.format(i)]) + \
                          self.__weights['b{}'.format(i+1)]
            if i != self.__L - 1:
                if self.__activation == 'sig':
                    self.__cache['A{}'.format(i+1)] = sigmoid(z)
                else:
                    self.__cache['A{}'.format(i + 1)] = tanh(z)
            else:
                t = np.exp(z)
                self.__cache['A{}'.format(self.__L)] = t * np.sum(t)
        return self.__cache['A{}'.format(self.__L)], self.__cache

    def cost(self, Y, A):
        """method"""
        m = Y.shape[1]
        return -np.sum(Y * np.log(A)) / m

    def evaluate(self, X, Y):
        """method"""
        A = self.forward_prop(X)[0]
        cost = self.cost(Y, A)
        A = np.where(A == np.amax(A, axis=0), 1, 0)
        return A, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """method"""
        m = Y.shape[1]
        copied = self.__weights.copy()
        for i in range(self.__L, 0, -1):
            if i == self.__L:
                dz = cache['A{}'.format(self.__L)] - Y
            else:
                dz = np.multiply(np.matmul(
                    copied['W{}'.format(i + 1)].T, dz),
                    cache['A{}'.format(i)] * (1 - cache['A{}'.format(i)]))
            dw = np.matmul(dz, cache['A{}'.format(i - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            self.__weights['W{}'.format(i)] = \
                self.__weights['W{}'.format(i)] - alpha*dw
            self.__weights['b{}'.format(i)] = \
                self.__weights['b{}'.format(i)] - alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """train"""
        if not type(iterations) is int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not type(alpha) is float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if not type(step) is int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        cost = []
        iter = []
        i = 0
        for epoch in range(iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(Y, self.__cache, alpha)
            if epoch % step == 0 and verbose:
                cost.append(self.cost(Y, self.__cache['A{}'.format(self.__L)]))
                iter.append(epoch)
                print('Cost after {} iterations: {}'.format(epoch, cost[i]))
                i += 1
            if graph:
                plt.plot(iter, cost)
                plt.xlabel("iterations")
                plt.ylabel("cost")
                plt.title("Training cost")
        return self.evaluate(X, Y)

    def save(self, filename):
        """method"""
        if '.pkl' not in filename:
            filename = filename + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """method"""
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError as e:
            return None
