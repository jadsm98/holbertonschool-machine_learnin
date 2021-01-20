#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def l2_reg_cost(cost):
    """function"""
    return cost + tf.contrib.layers.l2_regularizer(0.1)
