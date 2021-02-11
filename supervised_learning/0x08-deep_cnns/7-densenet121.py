#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """function"""
    w = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    BN = K.layers.BatchNormalization()(X)
    act = K.layers.Activation('relu')(BN)
    conv = K.layers.Conv2D(64, kernel_size=7, padding='same',
                           strides=(2, 2), kernel_initializer=w)(act)
    maxpool = K.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2),
                                 padding='same')(conv)
    dense1 = dense_block(maxpool, maxpool.shape[-1], growth_rate, 6)
    trans1 = transition_layer(dense1[0][-1], dense1[1], compression)
    dense2 = dense_block(trans1[0], trans1[1], growth_rate, 12)
    trans2 = transition_layer(dense2[0][-1], dense2[1], compression)
    dense3 = dense_block(trans2[0], trans2[1], growth_rate, 24)
    trans3 = transition_layer(dense3[0][-1], dense3[1], compression)
    dense4 = dense_block(trans3[0], trans3[1], growth_rate, 16)
    avgepool = K.layers.AvgPool2D(pool_size=(7, 7),
                                  strides=(1, 1))(dense4[0][-1])
    soft = K.layers.Dense(1000, activation='softmax',
                          kernel_initializer=w)(avgepool)
    model = K.Model(inputs=X, outputs=soft)
    return model
