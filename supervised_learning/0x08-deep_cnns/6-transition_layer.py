#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """function"""
    w = K.initializers.he_normal()
    BN = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(BN)
    conv = K.layers.Conv2D(int(compression*nb_filters), kernel_size=1,
                           kernel_initializer=w)(act)
    avgepool = K.layers.AvgPool2D(pool_size=(2, 2), padding='same')(conv)
    return avgepool, avgepool.shape[-1]
