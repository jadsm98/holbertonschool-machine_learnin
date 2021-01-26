#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, verbose=True, shuffle=False):
    """function"""
    if early_stopping and not validation_data is None:
        early_stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience)
        callbacks = [early_stop]
    if learning_rate_decay and not validation_data is None:
        l_rate = K.optimizers.schedules.InverseTimeDecay(
            alpha, decay_steps=1, decay_rate=decay_rate)
        learning_rate = K.callbacks.LearningRateScheduler(
            l_rate, verbose=1)
        callbacks.append(learning_rate)
    else:
        callbacks = None
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          callbacks=callbacks, shuffle=shuffle,
                          validation_data=validation_data)
    return history
