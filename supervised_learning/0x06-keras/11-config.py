#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def save_config(network, filename):
    """function"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """function"""
    with open(filename, 'r'0 as f:
        read = f.read()
    from_json = K.models.model_from_json(read)
    return from_json
