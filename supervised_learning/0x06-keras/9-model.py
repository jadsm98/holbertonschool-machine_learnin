#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def save_model(network, filename):
    """function"""
    network.save(filename)
    return None


def load_model(filename):
    """function"""
    return K.models.load_model(filename)
