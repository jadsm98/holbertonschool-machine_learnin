#!/usr/bin/env python3
"""
module
"""


class Poisson:
    """
    class
    """

    def __init__(self, data=None, lambtha=1.):
        """
        constructor
        """
        if data is None:
           self.lambtha = float(lambtha)
           if self.lambtha <= 0:
               raise ValueError('lambtha must be a positive value')
        else:
            if not type(data) is list:
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """Calculates the value of the PMF."""
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        k_factorial = 1
        if k != 0:
            for i in range(2, k+1):
                k_factorial = k_factorial * i
        return ((self.lambtha ** (k)) *
                (2.7182818285 ** (-(self.lambtha)))) / k_factorial

    def cdf(self, k):
        """Calculates the value of the CDF."""
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        CDF = 0
        for i in range(0, k+1):
            CDF = CDF + self.pmf(i)
        return CDF
