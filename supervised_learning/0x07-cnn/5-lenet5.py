#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


def lenet5(X):
    """function"""
    weights = tf.keras.initializers.he_normal()
    conv1 = K.layers.Conv2D(6, kernel_size=5, padding='same',
                            activation='relu',
                            kernel_initializer=weights)(X)
    pool1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(16, kernel_size=5, padding='valid',
                            activation='relu',
                            kernel_initializer=weights)(pool1)
    pool2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2))(conv2)
    flatten = K.layers.Flatten()(pool2)
    dense1 = K.layers.Dense(120, activation='relu',
                            kernel_initializer=weights)(flatten)
    dense2 = K.layers.Dense(84, activation='relu',
                            kernel_initializer=weights)(dense1)
    Y = K.layers.Dense(10, activation='softmax',
                            kernel_initializer=weights)(dense2)
    model = K.Model(inputs=X, outputs=Y)
    opt = K.optimizers.Adam()
    model.compile(opt, metrics=['accuracy'])
    return model
