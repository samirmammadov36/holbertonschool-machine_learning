#!/usr/bin/env python3
"""Sort a DataFrame by High price in descending order."""


def high(df):
    """Return DataFrame sorted by High column descending."""
    return df.sort_values(by="High", ascending=False)
