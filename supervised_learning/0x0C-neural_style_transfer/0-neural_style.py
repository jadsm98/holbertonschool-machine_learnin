#!/usr/bin/env python3
"""module"""


import tensorflow as tf
import numpy as np


class NST:
    """class"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """constructor"""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3:
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')
            if not isinstance(content_image, np.ndarray) or content_image.ndim != 3:
            raise TypeError('content_image must be a numpy.ndarray with shape (h, w, 3)')
        if alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if beta < 0:
            raise TypeError('beta must be a non-negative number')
        tf.executing_eagerly()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
