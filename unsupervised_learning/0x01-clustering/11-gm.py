#!/usr/bin/env python3
"""module"""

import sklearn.mixture


def gmm(X, k):
  """function"""
  gm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
  return gm.weights_, gm.means_, gm.covariances_,
         gm.predict(X), gm.bic(X)
