#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """function"""
    results = network.evaluate(data, labels,
                               verbose=int(verbose))
    return results
