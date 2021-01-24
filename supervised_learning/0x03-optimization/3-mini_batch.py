#!/usr/bin/env python3

"""function"""


import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid,
                     Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """function"""
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]
        m = X_train.shape[0]
        if m % batch_size == 0:
            batches = m / batch_size
            extra = False
        else:
            batches = m // batch_size
            extra = True
        for i in range(epochs + 1):
            cost_tr, acc_tr = sess.run([loss, accuracy],
                                       feed_dict={x: X_train, y: Y_train})
            cost_val, acc_val = sess.run([loss, accuracy],
                                         feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_tr))
            print("\tTraining Accuracy: {}".format(acc_tr))
            print("\tValidation Cost: {}".format(cost_val))
            print("\tValidation accuracy: {}".format(acc_val))
            X_t, Y_t = shuffle_data(X_train, Y_train)
            if i != epochs + 1:
                for batch in range(batches + 1):
                    if batch == batches and extra:
                        X_b = X_t[batch * batch_size:m]
                        Y_b = Y_t[batch * batch_size:m]
                    else:
                        if not extra and batch == batches:
                            continue
                        X_b = X_t[batch * batch_size:(batch + 1)* batch_size]
                        Y_b = Y_t[batch * batch_size:(batch + 1)* batch_size]
                    sess.run([train_op], feed_dict={x:X_b, y:Y_b})
                    if (batch + 1) % 100 == 0 and batch != 0:
                        cost_b, acc_b = sess.run([loss, accuracy],
                                                 feed_dict={x:X_b, y:Y_b})
                        print("\tStep: {}".format(batch + 1))
                        print("\t\tCost: {}".format(cost_b))
                        print("\t\tAccuracy: {}".format(acc_b))
        return saver.save(sess, save_path)
