#!/usr/bin/env python3
"""module"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function"""
    m = images.shape[0]
    in_h = images.shape[1]
    in_w = images.shape[2]
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]
    out_h = in_h - k_h + 1
    out_w = in_w - k_w + 1
    output = np.zeros((m, out_h, out_w))
    for y in range(out_h):
        for x in range(out_w):
            im_slice = images[:, y:y + k_h, x:x + k_w]
            output[:, y, x] = np.sum(kernel * im_slice, axis=(1, 2))
    return output
