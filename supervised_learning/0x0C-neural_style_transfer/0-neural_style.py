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

        @staticmethod
        def scale_image(image):
            """method"""
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')
        h, w, _ = image.shape
        if h == max(h, w):
            new_size = (512, int(512*w/h))
        else:
            new_size = (int(512*h/w), 512)
        inter = tf.image.ResizeMethod.BICUBIC
        resized = tf.image.resize(image, new_size, method=inter)
        h_new, w_new = new_size
        reshape = tf.reshape(resized, (1, h_new, w_new, 3))
        scaled = tf.divide(reshape, 255)
        return scaled
