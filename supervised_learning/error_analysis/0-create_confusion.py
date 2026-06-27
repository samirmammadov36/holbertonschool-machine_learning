#!/usr/bin/env python3
"""Module for creating a confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Create a confusion matrix from labels and predictions."""
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    correct_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    for correct, predicted in zip(correct_labels, predicted_labels):
        confusion[correct, predicted] += 1

    return confusion
