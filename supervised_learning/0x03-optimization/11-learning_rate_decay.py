#!/usr/bin/env python3
"""module"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """function"""
    dc = global_step // decay_step
    return alpha / (1 + decay_rate * dc)
