#!/usr/bin/env python3
"""Module for calculating precision."""

import numpy as np


def precision(confusion):
    """
    Calculate the precision for each class.

    Args:
        confusion: A confusion matrix where rows represent correct labels
            and columns represent predicted labels.

    Returns:
        A numpy.ndarray containing the precision of each class.
    """
    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    return true_positives / predicted_positives
