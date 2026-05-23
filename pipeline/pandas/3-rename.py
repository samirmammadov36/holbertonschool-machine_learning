#!/usr/bin/env python3
"""Rename and format DataFrame columns."""

import pandas as pd


def rename(df):
    """Rename Timestamp column to Datetime and format data."""
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    return df[["Datetime", "Close"]]
