#!/usr/bin/env python3
"""module"""

import tensorflow.keras as keras
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function"""
    inp1 = keras.Input(shape=(input_dims,))
    x = inp1
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
    mean = keras.layers.Dense(latent_dims, name="mean")(x)
    var = keras.layers.Dense(latent_dims, name="log_var")(x)
    eps = keras.backend.random_normal(shape=(tf.shape(mean)[0], tf.shape(var)[1]))
    Z = mean + tf.exp(0.5 * var) * eps
    encoder = keras.Model(inputs=inp1, outputs=[Z, mean, var])

    inp2 = keras.Input(shape=(latent_dims,))
    x = inp2
    for layer in hidden_layers[::-1]:
        x = keras.layers.Dense(layer, activation="relu")(x)
    x = keras.layers.Dense(input_dims, activation="sigmoid")(x)
    decoder = keras.Model(inputs=inp2, outputs=x)
    z, m ,v = encoder(inp1)
    out = decoder(z)
    auto = keras.Model(inputs=inp1, outputs=out)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
