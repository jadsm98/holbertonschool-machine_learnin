#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """function"""
    weights = K.initializers.he_normal()
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(64, kernel_size=7, activation='relu',
                            strides=(2, 2), kernel_initializer=weights,
                            padding='same')(X)
    BN = K.layers.BatchNormalization()(conv1)
    act = K.layers.Activation('relu')(BN)
    maxpool = K.layers.MaxPool2D(pool_size=(3, 3), padding='same',
                                 strides=(2, 2))(act)
    proj_block0 = projection_block(maxpool, [64, 64, 256], s=1)
    id_block2 = identity_block(proj_block0, [64, 64, 256])
    id_block3 = identity_block(id_block2, [64, 64, 256])

    proj_block1 = projection_block(id_block3, [128, 128, 512])
    id_block4 = identity_block(proj_block1, [128, 128, 512])
    id_block5 = identity_block(id_block4, [128, 128, 512])
    id_block6 = identity_block(id_block5, [128, 128, 512])

    proj_block2 = projection_block(id_block6, [256, 256, 1024])
    id_block7 = identity_block(proj_block2, [256, 256, 1024])
    id_block8 = identity_block(id_block7, [256, 256, 1024])
    id_block9 = identity_block(id_block8, [256, 256, 1024])
    id_block10 = identity_block(id_block9, [256, 256, 1024])
    id_block11 = identity_block(id_block10, [256, 256, 1024])

    proj_block3 = projection_block(id_block11, [512, 512, 2048])
    id_block12 = identity_block(proj_block3, [512, 512, 2048])
    id_block13 = identity_block(id_block12, [512, 512, 2048])

    avgepool = K.layers.AvgPool2D(pool_size=(7, 7), strides=(1, 1))(id_block13)
    soft = K.layers.Dense(1000, activation='softmax',
                          kernel_initializer=weights)(avgepool)
    model = K.Model(inputs=X, outputs=soft)
    return model
