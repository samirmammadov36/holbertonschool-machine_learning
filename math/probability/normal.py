#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the normal distribution.

        Args:
            data (list): data used to estimate distribution
            mean (float): mean of the distribution
            stddev (float): standard deviation

        Raises:
            TypeError: if data is not a list
            ValueError: if data has fewer than 2 values
            ValueError: if stddev is not positive
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
