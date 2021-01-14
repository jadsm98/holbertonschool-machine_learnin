#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """function"""
    x = tf.placeholder("float", [None, nx], 'x')
    y = tf.placeholder("float", [None, classes], 'y')
    return x, y
