#!/usr/bin/env python3
"""Module for calculating specificity from a confusion matrix."""

import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix where rows represent
            actual labels and columns represent predicted labels.

    Returns:
        numpy.ndarray: Specificity value for each class.
    """
    true_positive = np.diag(confusion)
    false_positive = np.sum(confusion, axis=0) - true_positive
    false_negative = np.sum(confusion, axis=1) - true_positive
    true_negative = (
        np.sum(confusion)
        - true_positive
        - false_positive
        - false_negative
    )

    return true_negative / (true_negative + false_positive)
