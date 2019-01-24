"""
Utility functions for t-SNE meeting
"""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def conditional_probabilities(data, i, sigma=5.0):
    """
    Computes conditional neighborhood probabilities

    Inputs
        data (N x d): input data, samples in rows, features in columns
        i (scalar): index to compute probabilities against
        sigma (float): width of Gaussian to center over point i
    """

    n_samples = data.shape[0]

    distances = pairwise_distances(data, metric="euclidean", squared=True)
    p_jis = np.zeros((n_samples,))

    running_sum = 0.0
    for j in range(n_samples):
        if i == j:
            p_jis[j] = 0
        else:
            p_jis[j] = np.exp(-distances[i, j] / (2 * sigma ** 2))
            running_sum += p_jis[j]

    # normalization step
    p_jis /= running_sum
    p_jis = np.maximum(p_jis, 1e-16)

    return p_jis

def entropy(p_jis):
    """
    Computes entropy of distribution

    Inputs:
        p_jis (vec): vector to compute entropy of
    """
    n_samples = p_jis.shape[0]
    neg_entropy = 0.0

    for j in range(n_samples):
        neg_entropy += p_jis[j] * np.log(p_jis[j])
    
    return -neg_entropy
