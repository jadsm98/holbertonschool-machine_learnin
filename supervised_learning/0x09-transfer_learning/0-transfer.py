#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K


(X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()


def preprocess(X, Y):
    """function"""
    Y_p = K.utils.to_categorical(Y)
    X_p = K.applications.inception_v3.preprocess_input(X)
    return X_p, Y_p


X_train, Y_train = preprocess(X_train, Y_train)
X_valid, Y_valid = preprocess(X_valid, Y_valid)


base_model = K.applications.InceptionV3(include_top=False,
                                        input_shape=(224, 224, 3))
base_model.trainable = False

w = K.initializers.he_normal()
X = K.Input(shape=(32, 32, 3))
lamb = K.layers.Lambda(lambda x: K.backend.resize_images(x, 7, 7,
                       data_format='channels_last'))(X)
base = base_model(lamb, training=False)
avgepool = K.layers.GlobalAveragePooling2D()(base)
dense = K.layers.Dense(512, activation='relu',
                       kernel_initializer=w)(avgepool)
batch = K.layers.BatchNormalization()(dense)
drop = K.layers.Dropout(0.3)(batch)
soft = K.layers.Dense(10, activation='softmax',
                      kernel_initializer=w)(drop)
model = K.Model(inputs=X, outputs=soft)

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, epochs=5,
          validation_data=(X_valid, Y_valid))

model.save('cifar10.h5')
