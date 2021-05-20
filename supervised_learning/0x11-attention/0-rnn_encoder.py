#!/usr/bin/env python3
"""module"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True, return_state=True)

    def initialize_hidden_state(self):
        """method"""
        return tf.zeros(shape=(self.batch, self.units))

    def call(self, x, initial):
        """method"""
        out = self.embedding(x)
        output, hidden = self.gru(out, initial_state=initial)
        return output, hidden
