#!/usr/bin/env python3
"""Module"""

import tensorflow_hub as hub
import os
import numpy as np


def semantic_search(corpus_path, sentence):
    """Function"""

    documents = [sentence]

    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(corpus_path + '/' + filename,
                      mode='r', encoding='utf-8') as file:
                documents.append(file.read())
    embed = \
        hub.load("https://tfhub.dev/google/" +
                 "universal-sentence-encoder-large/5")
    embeddings = embed(documents)
    corr = np.inner(embeddings, embeddings)
    most_similar = np.argmax(corr[0, 1:])
    text = documents[most_similar + 1]
    return text
