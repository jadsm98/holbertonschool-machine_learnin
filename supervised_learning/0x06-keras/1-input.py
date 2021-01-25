#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function"""
    inputs = K.Input(shape=(nx,))
    l2 = K.regularizers.l2(lambtha)
    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=l2)(inputs)
    drop = K.layers.Dropout(1 - keep_prob)(x)
    for layer in range(1, len(layers) - 1):
        x =  K.layers.Dense(layers[layer], activation=activations[layer],
                            kernel_regularizer=l2)(drop)
        drop = K.layers.Dropout(1 - keep_prob)(x)
    output = K.layers.Dense(layers[-1], activation=activations[-1],
                            kernel_regularizer=l2)(drop)
    return K.Model(inputs=inputs, outputs=output)
