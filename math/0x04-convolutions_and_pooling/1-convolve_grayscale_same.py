#!/usr/bin/env python3
"""module"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """function"""
    m = images.shape[0]
    in_h = images.shape[1]
    in_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    out_h = in_h
    out_w = in_w
    output = np.zeros((images.shape[0], out_h, out_w))
    pad_h = max((out_h - 1) + k_h - in_h, 0)
    pad_w = max((out_w - 1) + k_w - in_w, 0)
    pad_t = pad_h // 2
    pad_b = pad_h - pad_t
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    image_padded = np.zeros((m, in_h + pad_h, in_w + pad_w))
    image_padded[:, pad_t: - pad_b, pad_l: - pad_r] = images
    for y in range(out_h):
        for x in range(out_w):
            im_slice = image_padded[:, y: y + k_h, x: x + k_w]
            output[:, y, x] = np.sum(kernel * im_slice, axis=(1, 2))
    return output
