#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None,
                early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """function"""
    callbacks = []
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(monitor="val_loss",
                                               patience=patience)
        callbacks.append(early_stop)
    if learning_rate_decay and validation_data is not None:
        def step_decay(epoch):
            return alpha / (1 + decay_rate * epoch)
        learning_rate = K.callbacks.LearningRateScheduler(
            step_decay, verbose=1)
        callbacks.append(learning_rate)
    else:
        callbacks = None
    history = network.fit(data, labels, batch_size=batch_size,
                          epochs=epochs, verbose=verbose,
                          callbacks=callbacks, shuffle=shuffle,
                          validation_data=validation_data)
    val_losses = history.history['val_loss']
    if val_losses[-1] == min(val_losses) and save_best:
        model.save(filepath)
    return history
