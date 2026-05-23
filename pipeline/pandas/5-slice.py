#!/usr/bin/env python3
"""Slice specific columns and rows from a DataFrame."""


def slice(df):
    """Return selected columns with every 60th row."""
    return df[["High", "Low", "Close", "Volume_(BTC)"]][::60]
