#!/usr/bin/env python3
"""module"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initializer"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape((ac_samples, 1))
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """method"""
        mue, sig = self.gp.predict(self.X_s)
        if self.minimize:
            f_best = np.amin(self.gp.Y)
            upper = - mue + f_best - self.xsi
            Z = np.where(sig > 0, upper/sig, 0)
        else:
            f_best = np.amax(self.gp.Y)
            upper = mue - f_best - self.xsi
            Z = np.where(sig > 0, upper / sig, 0)
        cdf = norm.cdf(Z)
        pdf = norm.pdf(Z)
        eq = upper * cdf + sig * pdf
        EI = np.where(sig > 0, eq, 0)
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
