#!/usr/bin/env python3
"""Module for calculating the F1 score of a confusion matrix."""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Calculate the F1 score for each class.

    Args:
        confusion (numpy.ndarray): Confusion matrix where rows represent
            actual labels and columns represent predicted labels.

    Returns:
        numpy.ndarray: F1 score for each class.
    """
    recall = sensitivity(confusion)
    precision_values = precision(confusion)

    return 2 * precision_values * recall / (precision_values + recall)
