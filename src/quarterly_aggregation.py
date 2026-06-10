"""Quarterly aggregation helpers for RateWall contract exports."""

from __future__ import annotations

import pandas as pd


def quarter_label(date_value) -> str:
    period = pd.Period(pd.to_datetime(date_value), freq="Q")
    return f"{period.year}Q{period.quarter}"


def add_quarter_column(df: pd.DataFrame, date_column: str = "Date") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["quarter"] = out[date_column].apply(quarter_label)
    return out

