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

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """method"""
        pick = []
        x1, y1, x2, y2 = filtered_boxes[:, 0], filtered_boxes[:, 1],\
            filtered_boxes[:, 2], filtered_boxes[:, 3]
        area = (x2 - x1)*(y2 - y1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            suppress = [last]
            for pos in range(0, last):
                j = idxs[pos]
                xx1 = max(x1[i], x1[j])
                yy1 = max(y1[i], y2[j])
                xx2 = min(x2[i], x2[j])
                yy2 = min(y2[i], y2[j])

                w = max(0, xx2 - xx1)
                h = max(0, yy2 - yy1)
                overlap = float(w*h)/area[j]
                if overlap > self.nms_t:
                    suppress.append(pos)
            idxs = np.delete(idxs, suppress)
        box_pred = filtered_boxes[pick]
        class_pred = box_classes[pick]
        scores_pred = box_scores[pick]
        return box_pred, class_pred, scores_pred

    @staticmethod
    def load_images(folder_path):
        """method"""
        images = glob.glob(folder_path + '/*')
        im = []
        for i in images:
            im.append(cv2.imread(i))
        return im, images

    def preprocess_images(self, images):
        """method"""
        new_width = self.model.input.shape[1]
        new_height = self.model.input.shape[2]
        new_im = []
        im_shape = []
        for im in images:
            im_shape.append(im.shape[0:2])
            resized = cv2.resize(im, (new_width, new_height),
                                 interpolation=cv2.INTER_CUBIC)
            rescaled = resized/255
            new_im.append(rescaled)
        pimage = np.array(new_im)
        image_shape = np.array(im_shape)
        return pimage, image_shape

        def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
            """method"""
            font = cv2.FONT_HERSHEY_SIMPLEX
            lt = cv2.LINE_AA
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                cv2.rectangle(image, (x1, y1), (x2, y2),
                              color=(255, 0, 0), thickness=2)
                class_name = self.class_names[box_classes[i]]
                score = box_scores[i]
                org = (x1, y1 - 5)
                cv2.putText(image, '{} {:0.2f}'.format(class_name, score), org=org,
                            fontFace=font, fontScale=0.5, thickness=1,
                            lineType=lt, color=(0, 0, 255))
            cv2.imshow(file_name, image)
            k = cv2.waitKey(0)
            if k == ord('s'):
                if not os.path.exists('detections'):
                    os.mkdir('detections')
                cv2.imwrite(os.path.join('detections', file_name), image)
                cv2.destroyAllWindows()
            else:
                cv2.destroyAllWindows()
