#!/usr/bin/env python3
"""module"""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """function"""

    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angle = (pos/np.power(10000, (2*(i//2)/np.float32(dm))))
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle
