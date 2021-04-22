#!/usr/bin/env python3
"""module"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """function"""
    l = len(hidden_layers) - 1
    encoder = keras.Sequential()
    for i, layer in enumerate(hidden_layers):
        if i == 0:
            encoder.add(keras.layers.Dense(layer, input_shape=(input_dims,),
                                           activation='relu'))
        else:
            encoder.add(keras.layers.Dense(layer, activation='relu'))
    encoder.add(keras.layers.Dense(latent_dims, activation='relu'))

    decoder = keras.Sequential()
    for i in range(l, -1, -1):
        if i == l:
            decoder.add(keras.layers.Dense(hidden_layers[i],
                                           input_shape=(latent_dims,),
                                           activation='relu'))
        else:
            decoder.add(keras.layers.Dense(hidden_layers[i],
                                           activation='relu'))
    decoder.add(keras.layers.Dense(input_dims, activation='sigmoid'))

    inp = keras.Input(shape=(input_dims,))
    x = encoder(inp)
    out = decoder(x)
    auto = keras.Model(inputs=inp, outputs=out)

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
