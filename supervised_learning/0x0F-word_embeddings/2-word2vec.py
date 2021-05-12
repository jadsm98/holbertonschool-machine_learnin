from gensim.models import Word2Vec
from gensim.test.utils import common_texts


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5,
                   cbow=True, iterations=5, seed=0, workers=1):
    """function"""

    model = Word2Vec(sentences=sentences, size=size, window=window,
                     workers=workers, seed=seed, negative=negative,
                     min_count=min_count, sg=int(not cbow))
    examples = model.corpus_count
    return model.train(corpus_iterable=sentences, total_examples=examples,
                       epochs=iterations)
