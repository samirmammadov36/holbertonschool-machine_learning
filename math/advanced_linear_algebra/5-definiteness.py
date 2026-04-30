#!/usr/bin/env python3
"""Definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """Calculates the definiteness of a matrix"""

    # Type check
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Valid matrix check
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if matrix.size == 0:
        return None

    # Must be symmetric
    if not np.allclose(matrix, matrix.T):
        return None

    # Eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Check definiteness
    if np.all(eigenvalues > 0):
        return "Positive definite"

    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    if np.all(eigenvalues < 0):
        return "Negative definite"

    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    if (np.any(eigenvalues > 0) and
            np.any(eigenvalues < 0)):
        return "Indefinite"

    return None
