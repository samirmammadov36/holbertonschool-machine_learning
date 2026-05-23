#!/usr/bin/env python3
"""Create a pandas DataFrame from a NumPy array."""

import pandas as pd


def from_numpy(array):
    """Create a DataFrame from a NumPy array with alphabetic columns."""
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
