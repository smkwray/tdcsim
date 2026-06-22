from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_tdcest import (
    build_selected_target_crosscheck,
    build_tdcest_crosscheck,
    summarize_selected_target_crosscheck,
    render_tdcest_crosscheck_markdown,
    summarize_tdcest_crosscheck,
)


def test_tdcest_crosscheck_matches_after_unit_conversion():
    replay = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "TDC_Change": [10.0, -5.0],
        }
    ).set_index("Date")
    tdcest = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "tdc_base_bank_only_ru_flow": [10_000.0, -5_000.0],
        }
    )

    crosscheck = build_tdcest_crosscheck(
        replay,
        tdcest,
        tdcest_columns=["tdc_base_bank_only_ru_flow"],
    )

    assert crosscheck["tdcest_value_bil"].tolist() == pytest.approx([10.0, -5.0])
    assert crosscheck["difference_bil"].tolist() == pytest.approx([0.0, 0.0])
    assert crosscheck["verdict"].tolist() == ["matched", "matched"]


def test_tdcest_crosscheck_labels_replay_tdc_placeholder_zero():
    replay = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-31"]),
            "TDC_Change": [0.0],
        }
    ).set_index("Date")
    tdcest = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-31"]),
            "tdc_base_bank_only_ru_flow": [10_000.0],
        }
    )

    crosscheck = build_tdcest_crosscheck(
        replay,
        tdcest,
        tdcest_columns=["tdc_base_bank_only_ru_flow"],
    )
    summary = summarize_tdcest_crosscheck(crosscheck)

    assert crosscheck.iloc[0]["verdict"] == "tdcsim_replay_tdc_not_implemented"
    assert crosscheck.iloc[0]["absolute_difference_bil"] == pytest.approx(10.0)
    assert summary.iloc[0]["tdcsim_replay_tdc_not_implemented_rows"] == 1
    assert summary.iloc[0]["max_abs_difference_bil"] == pytest.approx(10.0)
    assert "Replay TDC not implemented rows: 1" in render_tdcest_crosscheck_markdown(summary)


def test_tdcest_crosscheck_labels_missing_tdcest_rows():
    replay = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-31"]),
            "TDC_Change": [10.0],
        }
    ).set_index("Date")
    tdcest = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-06-30"]),
            "tdc_base_bank_only_ru_flow": [10_000.0],
        }
    )

    crosscheck = build_tdcest_crosscheck(
        replay,
        tdcest,
        tdcest_columns=["tdc_base_bank_only_ru_flow"],
    )

    assert crosscheck.iloc[0]["verdict"] == "no_tdcest_value"
    assert pd.isna(crosscheck.iloc[0]["tdcest_value_bil"])


def test_selected_target_crosscheck_matches_one_target_per_quarter():
    replay = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "TDC_Change": [10.0, -5.0],
        }
    ).set_index("Date")
    selected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-03-31", "2025-06-30"]),
            "quarter": ["2025Q1", "2025Q2"],
            "selected_tdc_value_mil": [10_000.0, -5_000.0],
            "selected_tdc_series_key": ["canonical", "canonical"],
            "replay_tdc_method_label": ["modern", "modern"],
            "replay_tdc_method_tier": ["constrained_component", "constrained_component"],
        }
    )

    crosscheck = build_selected_target_crosscheck(replay, selected)
    summary = summarize_selected_target_crosscheck(crosscheck)

    assert crosscheck["selected_tdc_value_bil"].tolist() == pytest.approx([10.0, -5.0])
    assert crosscheck["verdict"].tolist() == ["matched", "matched"]
    assert summary.iloc[0]["matched_rows"] == 2
