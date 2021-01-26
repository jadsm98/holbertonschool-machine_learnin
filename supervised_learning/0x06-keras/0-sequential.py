#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function"""
    model = K.models.Sequential()
    l2 = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(layers[0], activation=activations[0],
                             input_shape=(nx,), kernel_regularizer=l2))
    for layer in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[layer], activation=activations[layer],
                                 input_shape=(layers[layer-1],),
                                 kernel_regularizer=l2))
    return model
