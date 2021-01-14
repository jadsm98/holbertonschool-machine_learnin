#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """function"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    l = tf.layers.dense(units=n, activation=activation,
                        kernel_initializer=weights, name="layer")
    return l(prev)
