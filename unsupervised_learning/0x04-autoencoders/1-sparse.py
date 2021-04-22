#!/usr/bin/env python3
"""Module"""


import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """function"""
    l = len(hidden_layers) - 1
    inp1 = keras.Input((input_dims,))
    x = inp1
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation='relu')(x)
    L1 = keras.regularizers.l1(lambtha)
    x = keras.layers.Dense(latent_dims, activation='relu', activity_regularizer=L1)(x)
    encoder = keras.Model(inputs=inp1, outputs=x)
    inp2 = keras.Input((latent_dims,))
    x = inp2
    for layer in hidden_layers[::-1]:
        x = keras.layers.Dense(layer, activation='relu')(x)
    x = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=inp2, outputs=x)
    x = encoder(inp1)
    out = decoder(x)
    auto = keras.Model(inputs=inp1, outputs=out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
