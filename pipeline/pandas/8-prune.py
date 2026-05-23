#!/usr/bin/env python3
"""Remove rows with NaN values in the Close column."""


def prune(df):
    """Return DataFrame without NaN values in Close column."""
    return df.dropna(subset=["Close"])
