#!/usr/bin/env python3
"""Set Timestamp column as DataFrame index."""


def index(df):
    """Return DataFrame with Timestamp column set as index."""
    return df.set_index("Timestamp")
