#!/usr/bin/env python3
def cdf(self, k):
    """
    Calculates the CDF for a given number of successes.

    Args:
        k (int): number of successes

    Returns:
        float: CDF value for k
    """

    k = int(k)

    if k < 0:
        return 0

    cdf = 0

    for i in range(k + 1):

        factorial = 1
        for j in range(1, i + 1):
            factorial *= j

        pmf = ((self.e ** (-self.lambtha)) *
               (self.lambtha ** i)) / factorial

        cdf += pmf

    return cdf
