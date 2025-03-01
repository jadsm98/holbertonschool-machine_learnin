#!/usr/bin/env python3
"""
Gaussian Process module
"""

import numpy as np


class GaussianProcess():
    """
    Gaussian Process class
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializer method
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the Kernel
        """
        X = np.zeros((X1.shape[0], X2.shape[0]))
        for m in range(X1.shape[0]):
            for n in range(X2.shape[0]):
                X[m, n] = (X1[m] - X2[n])**2
        k = self.sigma_f**2 * np.exp(-X/(2*self.l**2))
        return k
