#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """function"""
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience)
        callbacks = [early_stop]
    else:
        callbacks = None
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          callbacks=callbacks, shuffle=shuffle,
                          validation_data=validation_data)
    return history
