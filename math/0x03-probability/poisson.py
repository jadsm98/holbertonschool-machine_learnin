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
