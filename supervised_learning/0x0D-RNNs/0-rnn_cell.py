#!/usr/bin/env python3
"""module"""


import numpy as np
from scipy.special import softmax


class RNNCell:
    """RNN cell class"""

    def __init__(self, i, h, o):
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """forward prop"""
        concat = np.hstack((h_prev, x_t))
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y = softmax(np.matmul(h_next, self.Wy) + self.by, axis=1)
        return h_next, y
  
