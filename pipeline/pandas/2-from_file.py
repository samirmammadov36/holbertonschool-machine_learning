#!/usr/bin/env python3
"""Load data from a file into a pandas DataFrame."""

import pandas as pd


def from_file(filename, delimiter):
    """Load data from a file as a pandas DataFrame."""
    return pd.read_csv(filename, delimiter=delimiter)
