#!/usr/bin/env python3
"""
module
"""


import numpy as np


def create_confusion_matrix(labels, logits):
    """function"""

    labels = list(labels)
    logits = list(logits)
    conf_matrix = [[0 for i in range(len(labels[0]))]
                   for i in range(len(labels[0]))]
    for i in range(len(labels)):
        for j in range(len(labels[0])):
            if labels[i][j] == 1:
                for k in range(len(logits[0])):
                    if logits[i][k] == 1:
                        conf_matrix[j][k] += 1
    return np.array(conf_matrix).astype('float64')
