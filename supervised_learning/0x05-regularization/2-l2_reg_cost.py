#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def l2_reg_cost(cost):
    """function"""
    return tf.math.add(cost, tf.losses.get_regularization_loss)
