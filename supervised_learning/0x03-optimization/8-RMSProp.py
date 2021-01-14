#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """function"""
    rms_op = tf.train.RMSPropOptimizer(alpha, decay=beta2, epsilon=epsilon)
    return rms_op.minimize(loss)
