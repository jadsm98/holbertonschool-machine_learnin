#!/usr/bin/env python3
"""module"""

import numpy as np


class GRUCell:
    """GRU cell class"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        """applies softmax activation"""
        return np.exp(z)/np.sum(np.exp(z), axis=1).reshape((-1, 1))

    @staticmethod
    def sigmoid(z):
        """applies sigmoid activation"""
        return 1/(1 + np.exp(-z))

    def forward(self, h_prev, x_t):
        """GRU forward prop"""

        concat1 = np.hstack((h_prev, x_t))
        z = self.sigmoid(np.matmul(concat1, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(concat1, self.Wr) + self.br)
        concat2 = np.hstack((r * h_prev, x_t))
        h = np.tanh(np.matmul(concat2, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
