#!/usr/bin/env python3
"""module"""

import numpy as np
from scipy.special import softmax, expit


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

    def forward(self, h_prev, x_t):
        """GRU forward prop"""

        concat1 = np.hstack((h_prev, x_t))
        z = expit(np.matmul(concat1, self.Wz) + self.bz)
        r = expit(np.matmul(concat1, self.Wr) + self.br)
        concat2 = np.hstack((r * h_prev, x_t))
        h = np.tanh(np.matmul(concat2, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h
        y = softmax(np.matmul(h_next, self.Wy) + self.by, axis=1)
        return h_next, y
