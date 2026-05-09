#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the normal distribution.
        """

        if data is None:

            if stddev <= 0:
                raise ValueError(
                    "stddev must be a positive value"
                )

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:

            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values"
                )

            self.mean = float(sum(data) / len(data))

            variance = 0

            for x in data:
                variance += (x - self.mean) ** 2

            variance /= len(data)

            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of x.
        """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a z-score.
        """

        return (z * self.stddev) + self.mean
