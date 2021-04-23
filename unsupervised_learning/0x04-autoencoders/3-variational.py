#!/usr/bin/env python3
"""module"""

import tensorflow.keras as keras


def Sampling(args):
    """Sampling"""
    mean, log_sig = args
    batch = keras.backend.shape(mean)[0]
    dim = keras.backend.int_shape(mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return mean + keras.backend.exp(log_sig / 2) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function"""
    inp1 = keras.Input(shape=(input_dims,))
    x = inp1
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
    mean = keras.layers.Dense(latent_dims, name="mean")(x)
    var = keras.layers.Dense(latent_dims, name="log_var")(x)
    Z = keras.layers.Lambda(Sampling)([mean, var])
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
