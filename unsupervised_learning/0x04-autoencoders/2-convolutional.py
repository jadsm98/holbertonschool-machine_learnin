#!/usr/bin/env python3
"""Module"""


import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """function"""
    inp1 = keras.Input(input_dims)
    x = inp1
    for filter in filters:
        x = keras.layers.Conv2D(filter, kernel_size=(3, 3),
                                padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D((2, 2), padding="same")(x)
    encoder = keras.Model(inputs=inp1, outputs=x)

    inp2 = keras.Input(latent_dims)
    x = inp2
    for filter in filters[::-1]:
        if filter == filters[0]:
            pad = "valid"
        else:
            pad = "same"
        x = keras.layers.Conv2D(filter, kernel_size=(3, 3),
                                padding=pad, activation="relu")(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[-1], (input_dims[0], input_dims[1]),
                            padding="same", activation="sigmoid")(x)
    decoder = keras.Model(inputs=inp2, outputs=x)
    x = encoder(inp1)
    out = decoder(x)
    auto = keras.Model(inputs=inp1, outputs=out)

    auto.compile(optimizer="adam", loss="binary_crossentropy")
    return encoder, decoder, auto
