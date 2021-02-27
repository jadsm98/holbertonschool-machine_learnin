#!/usr/bin/env python3
"""module"""


import tensorflow.keras as K
import numpy as np


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
    
    def sig(self, X):
        """function"""
        return 1/(1 + np.exp(-X))

    def process_outputs(self, outputs, image_size):
        """method"""
        im_h = image_size[0]
        im_w = image_size[1]
        boxes = []
        box_confidence = []
        box_class_prob = []
        for i in range(len(outputs)):
            gh, gw, na, c = outputs[i].shape
            boxes.append(np.zeros((gh, gw, na, 4)))
            box_confidence.append(np.zeros((gh, gw, na, 1)))
            box_class_prob.append(np.zeros((gh, gw, na, c - 5)))
            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]
            tx, ty = outputs[i][..., 0], outputs[i][..., 1]
            tw, th = outputs[i][..., 2], outputs[i][..., 3]
            cx, cy = np.arange(gw).reshape(1, gw, 1),\
                np.arange(gh).reshape(gh, 1, 1)
            bx, by = (self.sig(tx) + cx)/gw, (self.sig(ty) + cy)/gh
            bw, bh = pw*np.exp(tw)/self.model.input.shape[1].value,\
                ph*np.exp(th)/self.model.input.shape[2].value
            x1, y1, x2, y2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2
            boxes[i][..., 0] = x1*im_w
            boxes[i][..., 1] = y1*im_h
            boxes[i][..., 2] = x2*im_w
            boxes[i][..., 3] = y2*im_h
            box_class_prob[i] = self.sig(outputs[i][..., 5:])
            box_confidence[i] = self.sig(outputs[i][..., 4, np.newaxis])
        return boxes, box_confidence, box_class_prob
