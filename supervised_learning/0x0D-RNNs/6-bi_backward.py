#!/usr/bin/env python3
"""module"""

import numpy as np


class BidirectionalCell:
    """class"""

    def __init__(self, i, h, o):
        """constuctor"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(2*h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        """applies softmax activation"""
        return np.exp(z)/np.sum(np.exp(z), axis=1).reshape((-1, 1))

    def forward(self, h_prev, x_t):
        """forward prop"""
        concat = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """backward pass"""
        concat = np.hstack((h_next, x_t))
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev
