#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """function"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=weights, name="layer")
    out = layer(prev)
    drop = tf.layers.Dropout(1 - keep_prob)
    return drop(out)
