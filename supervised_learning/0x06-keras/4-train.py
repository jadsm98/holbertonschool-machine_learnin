#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs, verbose=True,
                shuffle=False):
    """function"""
    history = network.fit(x=data, y=labels, batch_size=batch_size,
                          epochs=epochs, verbose=int(verbose),
                          shuffle=shuffle)
    return K.callbacks.History
