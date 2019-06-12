"""
Utility functions
"""

import numpy as np
import tensorflow as tf

def norm_image(x):
    """
    Normalizes image to range [0, 1]
    """
    return (x - np.min(x))/np.ptp(x)

def total_variation_loss(image_tensor):
    """
    Total variation loss (TV)
    Higher TV indicates bigger differences between adjacent pixels
    """
    
    X = tf.squeeze(image_tensor)
    xdiff = X[1:, :, :] - X[:-1, :, :]
    ydiff = X[:, 1:, :] - X[:, :-1, :]

    xdiff_l2 = tf.sqrt(tf.reduce_sum(tf.square(xdiff)))
    ydiff_l2 = tf.sqrt(tf.reduce_sum(tf.square(ydiff)))
    
    tv_loss = xdiff_l2 + ydiff_l2
    return tv_loss
