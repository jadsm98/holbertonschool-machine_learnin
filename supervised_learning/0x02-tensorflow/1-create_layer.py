#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """function"""
    weights = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    return tf.layers.dense(prev, n, activation=activation,
                           kernel_initializer=weights, name="layer")
