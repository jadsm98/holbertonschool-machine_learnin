#!/usr/bin/env python3
"""module"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """function"""
    m, h, w, c = images.shape
    kh, kw = kernel.shape[0], kernel.shape[1]
    sh, sw = stride
    if padding == 'valid':
        out_h = int(np.ceil((h - kh + 1) / sh))
        out_w = int(np.ceil((w - kw + 1) / sw))
    elif padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h + 1) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w + 1) / 2))
        out_h = int(np.ceil((h + 2*ph - kh + 1) / sh))
        out_w = int(np.ceil((w + 2*pw - kw + 1) / sw))
        im_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)])
    else:
        ph, pw = padding
        out_h = int(np.ceil((h + 2*ph - kh + 1) / sh))
        out_w = int(np.ceil((w + 2*pw - kw + 1) / sw))
        im_padded = np.pad(images, [(0, 0), (ph, ph), (pw, pw), (0, 0)])
    output = np.zeros((m, out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            if padding == 'valid':
                im_slice = images[:, i*sh: i*sh + kh, j*sw: j*sw + kw, :]
            else:
                im_slice = im_padded[:, i*sh: i*sh + kh,
                                     j*sw: j*sw + kw, :]
            output[:, i, j] = np.sum(im_slice * kernel, axis=(1, 2, 3))
    return output
