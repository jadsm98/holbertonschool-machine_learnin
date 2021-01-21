#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def l2_reg_cost(cost):
    """function"""
    reg_losses = tf.losses.get_regularization_losses)
    return cost + sum(reg_losses)
