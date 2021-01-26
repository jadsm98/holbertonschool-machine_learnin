#!/usr/bin/env python3
"""function"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """function"""
    prediction = network.predict(data, verbose=int(verbose))
    return prediction
