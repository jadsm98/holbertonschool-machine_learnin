  #!/usr/bin/env python3
"""module"""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """Class"""
    def __init__(self):
        """Initialize"""
        datasets = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True)
        self.data_train = datasets['train']
        self.data_valid = datasets['validation']
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def tokenize_dataset(self, data):
        """Method"""
        pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((pt.numpy() for pt, en in data), 2**15)
        en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus((en.numpy() for pt, en in data), 2**15)
        return pt, en
      
    def encode(self, pt, en):
        """Method"""
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1
        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_index]
        return pt_tokens, en_tokens
