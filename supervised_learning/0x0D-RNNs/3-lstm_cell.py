#!/usr/bin/env python3
"""module"""


import numpy as np


class LSTMCell:
    """LSTM cell class"""

    def __init__(self, i, h, o):
        """constructor"""
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        """applies softmax activation"""
        return np.exp(z)/np.sum(np.exp(z), axis=1).reshape((-1, 1))

    @staticmethod
    def sigmoid(z):
        """applies sigmoid activation"""
        return 1/(1 + np.exp(-z))

    def forward(self, h_prev, c_prev, x_t):
        """LSTM forward prop"""
        combine = np.hstack((h_prev, x_t))
        f = self.sigmoid(np.matmul(combine, self.Wf) + self.bf)
        u = self.sigmoid(np.matmul(combine, self.Wu) + self.bu)
        c = np.tanh(np.matmul(combine, self.Wc) + self.bc)
        o = self.sigmoid(np.matmul(combine, self.Wo) + self.bo)
        c_next = f * c_prev + u * c
        h_next = np.tanh(c_next) * o
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y
