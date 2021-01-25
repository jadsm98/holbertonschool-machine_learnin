#!/usr/bin/env python3
"""module"""


def train_model(network, data, labels,
                batch_size, epochs, verbose=True,
                shuffle=False):
    """function"""
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          shuffle=shuffle)
    return history
