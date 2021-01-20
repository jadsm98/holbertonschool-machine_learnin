#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def l2_reg_cost(cost):
    """function"""
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    weights = tf.get_variable(initializer=init, regularizer=regularizer)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_variables)
    return cost + reg_term
