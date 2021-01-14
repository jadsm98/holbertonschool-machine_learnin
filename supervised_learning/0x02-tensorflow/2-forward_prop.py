#!/usr/bin/env python3
"""module"""


import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """function"""
    for layer in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[layer], activations[layer])
    return x
