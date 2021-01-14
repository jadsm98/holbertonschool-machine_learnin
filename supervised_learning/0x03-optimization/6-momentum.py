#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """function"""
    return tf.train.MomentumOptimizer(alpha, beta).minimize(loss)
