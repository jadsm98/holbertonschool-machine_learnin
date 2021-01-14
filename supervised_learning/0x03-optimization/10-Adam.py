#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """function"""
    adam_op = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam_op.minimize(loss)
