#!/usr/bin/env python3
"""module"""


import numpy as np
import sklearn as sk


def bag_of_words(sentences, vocab=None):
    """function"""

    vector = sk.feature_extraction.text.CountVectorizer(vocabulary=vocab)
    emb = vector.fit_transform(sentences)
    embeddings = np.asarray(emb)
    features = vector.get_feature_names()
    return embeddings, features
