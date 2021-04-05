#!/usr/bin/env python3
"""module"""


import numpy as np


def regular(P):
    """function"""
    eig_val, _ = np.linalg.eig(P)
    if not np.all(eig_val <= 1) or not np.isreal(eig_val).all():
        return None
    try:
        
        n = P.shape[0]
        M = (P-np.identity(n)).T
        Q = np.vstack((M[:n - 1, :], np.ones((1, n))))
        Z = np.zeros((n, 1))
        Z[-1, :] = 1
        sol = np.linalg.solve(Q, Z).T
        return sol
    except Exception:
        return None
