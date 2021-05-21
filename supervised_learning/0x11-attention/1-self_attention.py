#!/usr/bin/env python3
"""module"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """class"""

    def __init__(self, units):
        """constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """method"""
        sequence = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(sequence) +
                                  self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        vec = tf.reduce_sum(weights*hidden_states, axis=1)
        return vec, weights
