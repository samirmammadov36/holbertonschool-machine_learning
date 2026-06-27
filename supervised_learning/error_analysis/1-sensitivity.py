#!/usr/bin/env python3
"""Module for calculating sensitivity."""

import numpy as np


def sensitivity(confusion):
    """
    Calculate the sensitivity for each class.

    Args:
        confusion: Confusion matrix where rows represent correct labels
            and columns represent predicted labels.

    Returns:
        A numpy.ndarray containing the sensitivity of each class.
    """
    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    return true_positives / actual_positives
