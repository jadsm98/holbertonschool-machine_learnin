#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def save_config(network, filename):
    """function"""
    network.to_json(filename)
    return None


def load_config(filename):
    """function"""
    from_json = model_from_json(filename)
    return from_json
