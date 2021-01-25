#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """function"""
    if early_stopping and not validation_data is None:
        early_stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience)
    else:
        early_stop = None
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose, callbacks=[early_stop],
                          shuffle=shuffle, validation_data=validation_data)
    return history
