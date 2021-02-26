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
            for h in range(gh):
                for w in range(gw):
                    for a in range(na):
                        pw = self.anchors[i, a, 0]
                        ph = self.anchors[i, a, 1]
                        tx, ty, tw, th = outputs[i][h, w, a, :4]
                        cx, cy = w, h
                        bx, by = (sig(tx) + cx)/gw, (sig(ty) + cy)/gh
                        bw, bh = pw*np.exp(tw)/self.model.input.shape[1],\
                            ph*np.exp(th)/self.model.input.shape[2]
                        x1, y1, x2, y2 = bx - bw/2, by - bh/2, bx + bw/2, by + bh/2
                        boxes[i][h, w, a, :] = x1*im_w, y1*im_h, x2*im_w, y2*im_h
                        box_class_prob[i][h, w, a, :] = sig(outputs[i][h, w, a, 5:])
                        box_confidence[i][h, w, a, :] = sig(outputs[i][h, w, a, 4])
        return boxes, box_confidence, box_class_prob

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """method"""
        box = []
        classes = []
        scores =[]
        for i in range(len(boxes)):
            boxScore = box_confidences[i] * box_class_probs[i]
            boxClass = np.argmax(boxScore, axis=-1)
            boxScore = np.max(boxScore, axis=-1)
            box.append(boxes[i][boxScore >= self.class_t])
            scores.append(boxScore[boxScore >= self.class_t])
            classes.append(boxClass[boxScore >= self.class_t])
        box = np.concatenate(box, axis=0)
        classes = np.concatenate(classes, axis=None)
        scores = np.concatenate(scores, axis=None)
        return box, classes, scores
