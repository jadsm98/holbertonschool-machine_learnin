#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def l2_reg_cost(cost):
    """function"""
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    weights = tf.Variable(initial_value=init)
    L2 = tf.contrib.layers.l2_regularizer(0.1)
    return cost + L2(weights)
