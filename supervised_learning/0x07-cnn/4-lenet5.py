#!/usr/bin/env python3
"""module"""


import tensorflow as tf


def lenet5(x, y):
    """function"""
    relu = tf.nn.relu
    weights = tf.contrib.layers.variance_scaling_initializer()
    conv1 = tf.layers.Conv2D(6, kernel_size=5, padding='same',
                             activation=relu, kernel_initializer=weights)
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    conv2 = tf.layers.Conv2D(16, kernel_size=5, padding='valid',
                             activation=relu, kernel_initializer=weights)
    pool2 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    flatten = tf.layers.Flatten()
    dense1 = tf.layers.Dense(120, activation=relu,
                             kernel_initializer=weights)
    dense2 = tf.layers.Dense(84, activation=relu,
                             kernel_initializer=weights)
    dense3 = tf.layers.Dense(10, kernel_initializer=weights)
    y_pred = dense3(dense2(dense1(flatten(pool2(conv2(pool1(conv1(x))))))))

    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    prediction = tf.math.argmax(y_pred, axis=1)
    correct = tf.math.argmax(y, axis=1)
    equality = tf.math.equal(prediction, correct)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    train_op = tf.train.AdamOptimizer().minimize(loss)
    return tf.nn.softmax(y_pred), train_op, loss, accuracy
