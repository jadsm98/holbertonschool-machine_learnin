#!/usr/bin/env python3
"""module"""

GP = __import__('2-gp').GaussianProcess
import numpy as np


class BayesianOptimization:
    """class"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initializer"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape((ac_samples, 1))
        self.xsi = xsi
        self.minimize = minimize
