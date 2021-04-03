#!/usr/bin/env python3
"""module"""

import sklearn.mixture


def gmm(X, k):
    """function"""
    gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi, m, S = gm.weights_, gm.means_, gm.covariances_
    clss, bic = gm.predict(X), gm.bic(X)
    return pi, m, S, clss, bic
