#!/usr/bin/env python3
"""module"""
  
  
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """function"""
    
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not type(kmin) is int or kmin <= 0 or kmin >= X.shape[0]:
        return None, None, None, None
    if not type(kmax) is int or kmax <= 0 or kmax >= X.shape[0]:
        return None, None, None, None
    if kmin >= kmax:
        return None, None, None, None
    if not type(iterations) is int or iterations <= 0:
        return None, None, None, None
    if not type(tol) is float or tol <= 0:
        return None, None, None, None
    if not type(verbose) is bool:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    bic_list = []
    log_like = []
    best = kmin
    for i in range(kmin, kmax + 1):
        pi, m, S, _, l = expectation_maximization(X, k, iterations,
                                                  tol, verbose)
        bic = (6*i - 1) * np.log(X.shape[0]) - 2*l
        if bic[best - kmin] > bic:
            best = i
            res = (pi, m, S)
        bic_list.append(bic)
        log_like.append(l)
    return best, res, np.asarray(log_like), np.asarray(bic_list)
