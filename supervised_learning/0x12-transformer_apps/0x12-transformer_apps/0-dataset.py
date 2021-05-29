#!/usr/bin/env python3
"""module"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class"""
    def __init__(self):
        """Initializor"""
        datasets = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True)
        self.data_train = datasets['train']
        self.data_valid = datasets['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """tokenizers"""
        pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in data), 2**15)
        en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in data), 2**15)
        return pt, en
