#!/usr/bin/env python3
"""module"""


import sklearn.cluster


def kmeans(X, k):
    """function"""
    kmean = sklearn.cluster.KMeans(n_clusters=k).fit(x)
    return kmean.cluster_centers_, kmean.labels_
