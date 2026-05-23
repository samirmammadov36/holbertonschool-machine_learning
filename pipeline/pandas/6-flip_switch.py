#!/usr/bin/env python3
"""Sort DataFrame in reverse chronological order and transpose it."""


def flip_switch(df):
    """Reverse the DataFrame rows and transpose it."""
    return df.sort_index(ascending=False).T
