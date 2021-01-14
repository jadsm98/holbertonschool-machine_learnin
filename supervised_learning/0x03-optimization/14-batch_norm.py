#!/usr/bin/env python3
"""module""" 


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """function"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layers = tf.layers.Dense(units=n, kernel_initializer=init)
    z = layers(prev)
    gamma = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                        trainable=True)
    beta = tf.Variable(initial_value=tf.constant(1.0, shape=[n]),
                       trainable=True)
    mean, var = tf.nn.moments(z, axes=[0])
    norm = tf.nn.batch_normalization(z, mean, var, offset=beta,
                                     scale=gamma, variance_epsilon=1e-8)
    if activation is None:
        return norm
    return activation(norm)
