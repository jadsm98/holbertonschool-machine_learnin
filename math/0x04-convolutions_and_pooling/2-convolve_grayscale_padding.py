#!/usr/bin/env python3
"""module"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function"""
    m, in_h, in_w, = images.shape
    k_h, k_w = kernel.shape
    p_h, p_w = padding
    out_w = in_w - k_w + 2*p_w + 1
    out_h = in_h - k_h + 2*p_h + 1
    output = np.zeros((m, out_h, out_w))
    image_padded = np.pad(images, [(0, 0), (p_h, p_h),
                                   (p_w, p_w)])
    for y in range(out_h):
        for x in range(out_w):
            im_slice = image_padded[:, y: y + k_h, x: x + k_w]
            output[:, y, x] = np.sum(kernel * im_slice, axis=(1, 2))
    return output
