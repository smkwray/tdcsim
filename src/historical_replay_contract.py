"""Contracts and validation helpers for historical replay loaders."""

from __future__ import annotations

import re

import pandas as pd


VALID_EVIDENCE_LABELS = frozenset(
    {"observed", "synthetic", "assumption", "residual", "not_available"}
)

QUARTERLY_CASH_REQUIRED_COLUMNS = (
    "quarter",
    "toc",
    "tga",
    "ttl",
    "toc_source_status",
    "tga_source_status",
    "ttl_source_status",
    "evidence_label",
)

SECTOR_POSITION_REQUIRED_COLUMNS = (
    "quarter",
    "sector",
    "measure",
    "value",
    "source_status",
    "evidence_label",
)

AUCTION_FLOW_REQUIRED_COLUMNS = (
    "quarter",
    "auction_id",
    "cusip",
    "security_type",
    "auction_date",
    "issue_date",
    "maturity_date",
    "offering_amount",
    "source_status",
    "evidence_label",
)

MSPD_COHORT_REQUIRED_COLUMNS = (
    "quarter",
    "cohort_id",
    "security_type",
    "issue_date",
    "maturity_date",
    "outstanding",
    "source_status",
    "evidence_label",
)


_QUARTER_RE = re.compile(r"^\d{4}Q[1-4]$")
_NUMERIC_NULL_MARKERS = frozenset({"", "null", "none", "nan", "na", "n/a", "*"})


def normalize_quarter_value(value) -> str:
    """Return a canonical YYYYQn quarter label."""

    if pd.isna(value):
        raise ValueError("quarter value is missing")
    text = str(value).strip()
    if not text:
        raise ValueError("quarter value is blank")
    if _QUARTER_RE.fullmatch(text):
        return text
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"invalid quarter value: {value!r}")
    return f"{ts.year}Q{ts.quarter}"


def normalize_quarter_series(values: pd.Series) -> pd.Series:
    """Normalize a Series of quarter-like values to YYYYQn strings."""

    return values.map(normalize_quarter_value)


def require_columns(
    frame: pd.DataFrame,
    required_columns: tuple[str, ...] | list[str] | set[str],
    *,
    dataset_name: str,
) -> None:
    """Fail loudly when a required schema column is absent."""

    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} missing required columns: {missing}")


def validate_duplicate_keys(
    frame: pd.DataFrame,
    key_columns: tuple[str, ...] | list[str],
    *,
    dataset_name: str,
) -> None:
    """Reject duplicated business keys in replay inputs."""

    require_columns(frame, key_columns, dataset_name=dataset_name)
    duplicates = frame.loc[frame.duplicated(subset=list(key_columns), keep=False), list(key_columns)]
    if duplicates.empty:
        return
    sample = duplicates.astype(str).drop_duplicates().head(3).to_dict(orient="records")
    raise ValueError(
        f"{dataset_name} has duplicate keys for {list(key_columns)}: {sample}"
    )


def filter_quarter_range(
    frame: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
    quarter_column: str = "quarter",
) -> pd.DataFrame:
    """Filter rows to an inclusive quarter range."""

    require_columns(frame, [quarter_column], dataset_name="quarter_range")
    quarter_periods = pd.PeriodIndex(normalize_quarter_series(frame[quarter_column]), freq="Q")
    result = frame.copy()
    if start_quarter is not None:
        start_period = pd.Period(normalize_quarter_value(start_quarter), freq="Q")
        result = result.loc[quarter_periods >= start_period].copy()
        quarter_periods = quarter_periods[quarter_periods >= start_period]
    if end_quarter is not None:
        end_period = pd.Period(normalize_quarter_value(end_quarter), freq="Q")
        result = result.loc[quarter_periods <= end_period].copy()
    return result.reset_index(drop=True)


def validate_numeric_columns(
    frame: pd.DataFrame,
    numeric_columns: tuple[str, ...] | list[str],
    *,
    dataset_name: str,
) -> pd.DataFrame:
    """Coerce numeric fields while rejecting non-numeric values."""

    result = frame.copy()
    for column in numeric_columns:
        if column not in result.columns:
            continue
        raw = result[column]
        source_null = raw.astype(str).str.strip().str.lower().isin(_NUMERIC_NULL_MARKERS)
        cleaned = raw.mask(source_null)
        converted = pd.to_numeric(cleaned, errors="coerce")
        invalid = converted.isna() & cleaned.notna() & cleaned.astype(str).str.strip().ne("")
        if invalid.any():
            bad_values = sorted(cleaned.loc[invalid].astype(str).unique().tolist())[:3]
            raise ValueError(f"{dataset_name} column {column} has non-numeric values: {bad_values}")
        result[column] = converted
    return result


def validate_evidence_labels(
    frame: pd.DataFrame,
    label_columns: tuple[str, ...] | list[str],
    *,
    dataset_name: str,
) -> None:
    """Check evidence labels against the replay contract vocabulary."""

    for column in label_columns:
        if column not in frame.columns:
            continue
        invalid = ~frame[column].astype(str).isin(VALID_EVIDENCE_LABELS)
        if invalid.any():
            bad_values = sorted(frame.loc[invalid, column].astype(str).unique().tolist())[:3]
            raise ValueError(
                f"{dataset_name} column {column} has invalid evidence labels: {bad_values}"
            )


def validate_source_status_columns(
    frame: pd.DataFrame,
    source_columns: tuple[str, ...] | list[str],
    *,
    dataset_name: str,
) -> None:
    """Require explicit source-status labels when a source-status column is present."""

    for column in source_columns:
        if column not in frame.columns:
            continue
        invalid = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
        if invalid.any():
            raise ValueError(f"{dataset_name} column {column} has blank source status labels")
