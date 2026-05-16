#!/usr/bin/env python3
"""Defines a Multivariate Normal distribution class."""

import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes a MultiNormal distribution.

        Args:
            data: numpy.ndarray of shape (d, n)
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        data_centered = data - self.mean

        self.cov = np.matmul(data_centered, data_centered.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Args:
            x: numpy.ndarray of shape (d, 1)

        Returns:
            float: PDF value
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        cov_det = np.linalg.det(self.cov)
        cov_inv = np.linalg.inv(self.cov)

        diff = x - self.mean

        exponent = -0.5 * np.matmul(
            np.matmul(diff.T, cov_inv),
            diff
        )[0][0]

        denominator = np.sqrt(((2 * np.pi) ** d) * cov_det)

        return (1 / denominator) * np.exp(exponent)
