#!/usr/bin/env python3
"""module"""


import numpy as np


def absorbing(P):
    """function"""

    if 1 in np.diag(P):
        n = np.count_nonzero(P == 1)
        Q = P[n:, n:]
        try:
            m = Q.shape[0]
            F = np.linalg.inv(np.eye(m) - Q)
            trial_num = np.sum(F, axis=1)
            if np.any(trial_num <= 0):
                return False
            else:
                return True
        except Exception:
            return False
    return False
