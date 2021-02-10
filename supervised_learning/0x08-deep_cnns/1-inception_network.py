#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """function"""
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, kernel_size=7, padding='same',
                            activation='relu', strides=(2, 2))(X)
    maxpool1 = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                                  strides=(2, 2))(conv1)
    conv2 = K.layers.Conv2D(64, kernel_size=1, padding='same',
                            activation='relu', strides=(1, 1))(maxpool1)
    conv3 = K.layers.Conv2D(192, kernel_size=3, padding='same',
                            activation='relu', strides=(1, 1))(conv2)
    maxpool2 = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                                  strides=(2, 2))(conv3)
    block1 = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    block2 = inception_block(block1, [128, 128, 192, 32, 96, 64])
    maxpool3 = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                                  strides=(2, 2))(block2)
    block3 = inception_block(maxpool3, [192, 96, 208, 16, 48, 64])
    block4 = inception_block(block3, [160, 112, 224, 24, 64, 64])
    block5 = inception_block(block4, [128, 128, 256, 24, 64, 64])
    block6 = inception_block(block5, [112, 144, 288, 32, 64, 64])
    block7 = inception_block(block6, [256, 160, 320, 32, 128, 128])
    maxpool4 = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                                  strides=(2, 2))(block7)
    block8 = inception_block(maxpool4, [256, 160, 320, 32, 128, 128])
    block9 = inception_block(block8, [384, 192, 384, 48, 128, 128])
    avgpool = K.layers.AvgPool2D(pool_size=(7, 7))(block9)
    drop = K.layers.Dropout(0.7)(avgpool)
    out = K.layers.Dense(1000, activation='softmax')(drop)
    model = K.Model(inputs=X, outputs=out)
    return model
