#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """function"""
    w = K.initializers.he_normal()
    output = [X]
    for i in range(layers):
        BN1 = K.layers.BatchNormalization()(X)
        act1 = K.layers.Activation('relu')(BN1)
        conv1 = K.layers.Conv2D(4*growth_rate, kernel_size=1,
                                padding='same', kernel_initializer=w)(act1)
        BN2 = K.layers.BatchNormalization()(conv1)
        act2 = K.layers.Activation('relu')(BN2)
        conv2 = K.layers.Conv2D(growth_rate, kernel_size=3,
                                padding='same', kernel_initializer=w)(act2)
        concat = K.layers.Concatenate()([X, conv2])
        output.append(concat)
        X = concat
    return output, output[-1].shape[-1]
