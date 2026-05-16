#!/usr/bin/env python3
"""Calculates the intersection for Bayesian probability."""

import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining the data
    with various hypothetical probabilities.

    Args:
        x: number of patients with severe side effects
        n: total number of patients observed
        P: hypothetical probabilities
        Pr: prior probabilities

    Returns:
        numpy.ndarray: intersection probabilities
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError(
            "Pr must be a numpy.ndarray with the same shape as P"
        )

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    factorial = np.math.factorial
    combination = factorial(n) / (factorial(x) * factorial(n - x))

    likelihood = combination * (P ** x) * ((1 - P) ** (n - x))

    return likelihood * Pr
