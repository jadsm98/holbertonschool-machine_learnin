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
    pad_h = max((out_h - 1) + k_h - in_h, 0) // 2
    pad_w = max((out_w - 1) + k_w - in_w, 0) // 2
    image_padded = np.pad(images, [(0, 0), (pad_h, pad_h),
                                   (pad_w, pad_w)])
    for y in range(out_h):
        for x in range(out_w):
            im_slice = image_padded[:, y: y + k_h, x: x + k_w]
            output[:, y, x] = np.sum(kernel * im_slice, axis=(1, 2))
    return output
