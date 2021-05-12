#!/usr/bin/env python3
"""module"""

from gensim.models import Word2Vec


def gensim_to_keras(model):
    """function"""
    keras_layer = model.wv.get_keras_embedding(train_embeddings=True)
    return keras_layer
