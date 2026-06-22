"""Tests for historical replay contract helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_contract import (
    filter_quarter_range,
    normalize_quarter_value,
    require_columns,
    validate_duplicate_keys,
    validate_evidence_labels,
    validate_numeric_columns,
)


def test_require_columns_raises_for_missing_schema():
    frame = pd.DataFrame({"quarter": ["2024Q1"]})

    with pytest.raises(ValueError, match="missing required columns"):
        require_columns(frame, ["quarter", "value"], dataset_name="contract_case")


def test_validate_duplicate_keys_raises_with_sample_rows():
    frame = pd.DataFrame(
        [
            {"quarter": "2024Q1", "sector": "banks", "measure": "level"},
            {"quarter": "2024Q1", "sector": "banks", "measure": "level"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate keys"):
        validate_duplicate_keys(
            frame,
            ["quarter", "sector", "measure"],
            dataset_name="sector_positions",
        )


def test_filter_quarter_range_is_inclusive():
    frame = pd.DataFrame({"quarter": ["2024Q1", "2024Q2", "2024Q3"], "value": [1, 2, 3]})

    result = filter_quarter_range(frame, start_quarter="2024Q2", end_quarter="2024Q3")

    assert result["quarter"].tolist() == ["2024Q2", "2024Q3"]


def test_validate_numeric_columns_rejects_non_numeric_values():
    frame = pd.DataFrame({"value": ["1.0", "bad"]})

    with pytest.raises(ValueError, match="non-numeric values"):
        validate_numeric_columns(frame, ["value"], dataset_name="numeric_case")


def test_validate_evidence_labels_rejects_unknown_labels():
    frame = pd.DataFrame({"evidence_label": ["observed", "guessed"]})

    with pytest.raises(ValueError, match="invalid evidence labels"):
        validate_evidence_labels(frame, ["evidence_label"], dataset_name="label_case")


def test_normalize_quarter_value_accepts_dates_and_quarter_labels():
    assert normalize_quarter_value("2024Q1") == "2024Q1"
    assert normalize_quarter_value("2024-06-30") == "2024Q2"
