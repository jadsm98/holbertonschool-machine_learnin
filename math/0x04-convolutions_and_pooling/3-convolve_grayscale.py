#!/usr/bin/env python3
"""module"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function"""
    m, in_h, in_w, = images.shape
    k_h, k_w = kernel.shape
    if padding == 'valid':
        out_h = (in_h - k_h + 1) // stride[0]
        out_w = (in_w - k_w + 1) // stride[1]
    elif padding == 'same':
        p_h = ((in_h - 1) * stride[0] + k_h - in_h) // 2
        p_w = ((in_w - 1) * stride[1] + k_w - in_w) // 2
        out_h = (in_h - k_h + 2 * p_h + 1) // stride[0]
        out_w = (in_w - k_w + 2 * p_w + 1) // stride[1]
        image_padded = np.pad(images, [(0, 0), (p_h, p_h),
                                       (p_w, p_w)])
    else:
        p_h, p_w = padding
        out_w = (in_w - k_w + 2 * p_w + 1) // stride[1]
        out_h = (in_h - k_h + 2 * p_h + 1) // stride[0]
        image_padded = np.pad(images, [(0, 0), (p_h, p_h),
                                       (p_w, p_w)])
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            j_s = j * stride[1]
            i_s = i * stride[0]
            if padding == 'valid':
                im_slice = images[:, i_s: i_s + k_h, j_s: j_s + k_w]
            else:
                im_slice = image_padded[:, i_s: i_s + k_h, j_s: j_s + k_w]
            output[:, i, j] = np.sum(kernel * im_slice, axis=(1, 2))
    return output
