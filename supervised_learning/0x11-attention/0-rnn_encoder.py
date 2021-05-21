#!/usr/bin/env python3
"""module"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """class"""
    def __init__(self, vocab, embedding, units, batch):
        """initializer"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, initial):
        """method"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden

    def initialize_hidden_state(self):
        """method"""
        return tf.zeros((self.batch, self.units))
