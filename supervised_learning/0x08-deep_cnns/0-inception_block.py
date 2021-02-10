#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """function"""
    conv1 = K.layers.Conv2D(filters[0], kernel_size=1, activation='relu')(A_prev)
    conv3R = K.layers.Conv2D(filters[1], kernel_size=1, activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(filters[2], kernel_size=3, padding='same', activation='relu')(conv3R)
    conv5R = K.layers.Conv2D(filters[3], kernel_size=1, activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(filters[4], kernel_size=5, padding='same', activation='relu')(conv5R)
    pool = K.layers.MaxPool2D(pool_size=(3, 3), padding='same', strides=(1, 1))(A_prev)
    convPP = K.layers.Conv2D(filters[5], kernel_size=1, activation='relu')(pool)
    out = K.layers.Concatenate()([conv1, conv3, conv5, convPP])
    return out
