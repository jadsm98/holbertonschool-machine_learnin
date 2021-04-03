#!/usr/bin/env python3
"""module"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """function"""
    
    
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not type(k) is int or k <=0 or X.shape[0] < k:
        return None, None, None, None, None
    if not type(iterations) is int or iterations <= 0:
        return None, None, None, None, None
    if not type(tol) is float or tol < 0:
        return None, None, None, None, None
    if not type(verbose) is bool:
        return None, None, None, None, None
    
    pi, m, S = initialize(X, k)
    loglike = 0
    for i in range(iterations):
        g, l = expectation(X, pi, m, S)
        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}"
                  .format(i, l.round(5)))
        if abs(l - loglike) < tol:
            if verbose:
                print("Log Likelihood after {} iterations: {}".format(
                    i, l.round(5)))
            return pi, m, S, g, l
        pi, m, S = maximization(X, g)
        loglike = l
    g, l = expectation(X, pi, m, S)
    if verbose:
        print("Log Likelihood after {} iterations: {}"
              .format(i, l.round(5)))
    return pi, m, S, g, l
