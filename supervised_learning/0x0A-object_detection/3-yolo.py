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

    def sig(self, x):
        """sigmoid"""
        return 1/(1 + np.exp(-x))
        
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """method"""
        score = []
        box = []
        for i in range(len(boxes)):
            box.append(boxes[i].reshape(-1, 4))
            score.append(box_confidences[i] * box_class_probs[i])
        boxClass = [np.argmax(i, axis=-1).reshape(-1) for i in score]
        boxScore = [np.max(i, axis=-1).reshape(-1) for i in score]
        box = np.concatenate(box)
        boxClass = np.concatenate(boxClass)
        boxScore = np.concatenate(boxScore)
        mask = np.where(boxScore >= self.class_t)
        filtered_boxes = box[mask]
        box_classes = boxClass[mask]
        box_scores = boxScore[mask]
        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """method"""
        box_filt = []
        classes_filt = []
        scores_filt = []
        for uniq in np.unique(box_classes):
            i = np.where(uniq == box_classes)
            box = filtered_boxes[i]
            classes = box_classes[i]
            scores = box_scores[i]

            x1, y1, x2, y2 = box[:, 0], box[:, 1],\
                box[:, 2], box[:, 3]
            area = (x2 - x1 + 1)*(y2 - y1 + 1)
            idxs = scores.argsort()[::-1]

            pick = []
            while len(idxs) > 0:
                i = idxs[0]
                pick.append(i)
                xx1 = np.maximum(x1[i], x1[idxs[1:]])
                yy1 = np.maximum(y1[i], y1[idxs[1:]])
                xx2 = np.minimum(x2[i], x2[idxs[1:]])
                yy2 = np.minimum(y2[i], y2[idxs[1:]])

                w = np.maximum(0, xx2 - xx1 + 1)
                h = np.maximum(0, yy2 - yy1 + 1)
                overlap = (w*h)/(area[i] + area[idxs[1:]] - w*h)
                idxs = idxs[np.where(overlap <= self.nms_t)[0] + 1]

            box_filt.append(box[pick])
            classes_filt.append(classes[pick])
            scores_filt.append(scores[pick])
        box_pred = np.concatenate(box_filt)
        classes_pred = np.concatenate(classes_filt)
        score_pred = np.concatenate(scores_filt)
        return box_pred, classes_pred, score_pred
