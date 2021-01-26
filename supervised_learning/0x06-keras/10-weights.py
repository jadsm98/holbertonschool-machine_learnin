#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """function"""
    network.save_weights(filename, save_format)
    return None


def load_weights(network, filename):
    """function"""
    network.load_weights(filename)
    return None
