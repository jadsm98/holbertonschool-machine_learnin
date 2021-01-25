#!/usr/bin/env python3
"""module"""


import tensorflow as tf
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """function"""
    model = K.models.Sequential()
    l2 = K.regularizers.l2(lambtha)
    for layer, nodes in enumerate(layers):
        model.add(K.layers.Dense(nodes, activation=activations[layer],
                                 input_shape=(nx,), kernel_regularizer=l2))
        if layer != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))
    return model
