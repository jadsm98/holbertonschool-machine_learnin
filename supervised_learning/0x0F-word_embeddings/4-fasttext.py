#!/usr/bin/env python3
"""module"""


from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """Function"""
    model = FastText(size=size, negative=negative,
                     window=window, seed=seed, cbow_mean=int(not cbow),
                     min_count=min_count, workers=workers)
    return model.train(sentences=sentences, total_examples=len(sentences),
                       epochs=iterations)
