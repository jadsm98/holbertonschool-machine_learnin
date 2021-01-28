#!/usr/bin/env python3
"""module"""


import numpy as np
from math import ceil


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function"""
    m, in_h, in_w, = images.shape
    k_h, k_w = kernel.shape
    if padding == 'valid':
        out_h = int(ceil(float(in_h - k_h + 1) / float(stride[0])))
        out_w = int(ceil(float(in_w - k_w + 1) / float(stride[1])))
    elif padding == 'same':
        out_h = int(ceil(float(in_h) / float(stride[0])))
        out_w = int(ceil(float(in_w) / float(stride[1])))
        p_h = max((out_h - 1) * stride[0] + k_h - in_h, 0)
        p_w = max((out_w - 1) * stride[1] + k_w - in_w, 0)
        p_t = p_h // 2
        p_b = p_h - p_t
        p_l = p_w // 2
        p_r = p_w - p_l
        image_padded = np.zeros((m, in_h + p_h, in_w + p_w))
        image_padded[:, p_t: - p_b, p_l: - p_r] = images
    else:
        p_h, p_w = padding
        out_w = int(ceil(float(in_w - k_w + 2 * p_w + 1) / float(stride[1])))
        out_h = int(ceil(float(in_h - k_h + 2 * p_h + 1) / float(stride[0])))
        image_padded = np.zeros((m, in_h + 2 * p_h, in_w + 2 * p_w))
        image_padded[:, p_h: - p_h, p_w: - p_w] = images
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
