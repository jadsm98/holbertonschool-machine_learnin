#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K
import numpy as np
import cv2
import glob


def sig(x):
    """sigmoid"""
    return 1/(1 + np.exp(-x))


class Yolo:
    """class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """initializer"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as file:
            self.class_names = file.read().splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
