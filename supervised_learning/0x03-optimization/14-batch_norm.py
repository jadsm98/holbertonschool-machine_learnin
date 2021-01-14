#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """function"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=k_init)
    z = layers(prev)
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]))
    beta = tf.Variable(initial_value=tf.constant(1.0, shape=[n]))
    norm = tf.nn.batch_normalization(Z, mean, var, offset=beta,
                                     scale=gamma, variance_epsilon=1e-8)
    return activation(norm)
