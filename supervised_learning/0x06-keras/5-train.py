#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                verbose=True, shuffle=False):
    """function"""
    if not type(validation_data) is tuple:
        validation_data = None
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data)
    return history
