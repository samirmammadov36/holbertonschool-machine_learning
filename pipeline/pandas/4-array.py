#!/usr/bin/env python3
"""Convert selected DataFrame values to a NumPy array."""


def array(df):
    """Return the last 10 rows of High and Close columns as ndarray."""
    return df[["High", "Close"]].tail(10).to_numpy()
