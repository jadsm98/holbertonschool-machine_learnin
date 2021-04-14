
#!/usr/bin/env python3
"""
Gaussian Process module
"""


import numpy as np


class GaussianProcess:
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

    def update(self, X_new, Y_new):
        """method"""
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
