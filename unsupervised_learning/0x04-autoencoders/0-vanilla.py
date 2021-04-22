#!/usr/bin/env python3
"""module"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function"""
    inp1 = keras.Input(shape=(input_dims,))
    x = inp1
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation="relu")(x)
    x = keras.layers.Dense(latent_dims, activation="relu")(x)
    encoder = keras.Model(inputs=inp1, outputs=x)
    inp2 = keras.Input(shape=(latent_dims,))
    x = inp2
    for layer in hidden_layers[::-1]:
        x = keras.layers.Dense(layer, activation="relu")(x)
    x = keras.layers.Dense(input_dims, activation="sigmoid")(x)
    decoder = keras.Model(inputs=inp2, outputs=x)
    x = encoder(inp1)
    out = decoder(x)
    auto = keras.Model(inputs=inp1, outputs=out)
    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
