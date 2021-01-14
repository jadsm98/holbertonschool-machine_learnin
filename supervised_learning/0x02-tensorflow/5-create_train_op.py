#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """function"""
    gradient = tf.train.GradientDescentOptimizer(alpha)
    return gradient.minimize(loss)
