from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_frn_rate_path
from frn_indexation import build_forward_frn_rate_path_rows, enrich_opening_frn_from_daily_indexes


def test_enrich_opening_frn_uses_latest_accrual_period_ending_at_opening_date(tmp_path: Path) -> None:
    frn_source = tmp_path / "frn_daily_indexes.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2026-06-17",
                "frn": "2-Year",
                "cusip": "912TESTFR",
                "original_issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "spread": 0.125,
                "start_of_accrual_period": "2026-06-20",
                "end_of_accrual_period": "2026-06-21",
                "daily_index": 3.625,
                "daily_int_accrual_rate": 3.750,
                "daily_accrued_int_per100": 0.010417,
                "accr_int_per100_pmt_period": 0.540000,
            },
            {
                "record_date": "2026-06-17",
                "frn": "2-Year",
                "cusip": "912TESTFR",
                "original_issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "spread": 0.125,
                "start_of_accrual_period": "2026-06-21",
                "end_of_accrual_period": "2026-06-22",
                "daily_index": 3.625,
                "daily_int_accrual_rate": 3.750,
                "daily_accrued_int_per100": 0.010417,
                "accr_int_per100_pmt_period": 0.550417,
            },
        ]
    ).to_csv(frn_source, index=False)
    portfolio = pd.DataFrame(
        [
            {
                "BondID": 1,
                "SecurityType": "FRN",
                "HolderType": "Private",
                "HolderSubBucket": "DomesticNonbank",
                "FaceValue": 200.0,
                "FixedSpread": 0.0,
                "BenchmarkRate_FRN": 0.0,
                "AccruedInterest_FRN": 0.0,
                "LastAccrualDate": pd.Timestamp("2025-01-31"),
                "IssueYieldAtIssue": 0.0,
                "_SourceCUSIP": "912TESTFR",
            }
        ]
    )

    enriched, metadata, diagnostics = enrich_opening_frn_from_daily_indexes(
        portfolio,
        frn_daily_indexes_path=frn_source,
        opening_date="2026-06-21",
        source_available_as_of="2026-06-17",
    )

    row = enriched.iloc[0]
    assert row["FixedSpread"] == pytest.approx(0.00125)
    assert row["BenchmarkRate_FRN"] == pytest.approx(0.03625)
    assert row["AccruedInterest_FRN"] == pytest.approx(1.08)
    assert row["LastAccrualDate"] == pd.Timestamp("2026-06-21")
    assert metadata["opening_frn_accrued_interest_bil"] == pytest.approx(1.08)
    assert metadata["frn_source_accrual_end_dates"] == ["2026-06-21"]
    assert diagnostics[0]["source_accrual_start"] == "2026-06-20"


def test_enrich_opening_frn_requires_complete_cusip_coverage(tmp_path: Path) -> None:
    frn_source = tmp_path / "frn_daily_indexes.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2026-06-17",
                "cusip": "912OTHERF",
                "original_issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "spread": 0.125,
                "start_of_accrual_period": "2026-06-20",
                "end_of_accrual_period": "2026-06-21",
                "daily_index": 3.625,
                "daily_int_accrual_rate": 3.750,
                "daily_accrued_int_per100": 0.010417,
                "accr_int_per100_pmt_period": 0.540000,
            }
        ]
    ).to_csv(frn_source, index=False)
    portfolio = pd.DataFrame(
        [
            {
                "BondID": 1,
                "SecurityType": "FRN",
                "HolderType": "Private",
                "HolderSubBucket": "",
                "FaceValue": 200.0,
                "_SourceCUSIP": "912TESTFR",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing 1 opening FRN CUSIPs"):
        enrich_opening_frn_from_daily_indexes(
            portfolio,
            frn_daily_indexes_path=frn_source,
            opening_date="2026-06-21",
            source_available_as_of="2026-06-17",
        )


def test_build_forward_frn_rate_path_uses_cbo_3m_decimal_surface(tmp_path: Path) -> None:
    rows = build_forward_frn_rate_path_rows(
        scenario_id="baseline",
        periods=[
            SimpleNamespace(period_start=pd.Timestamp("2026-09-20").date(), period_end=pd.Timestamp("2026-09-21").date())
        ],
        yield_surface_rows=[
            {"curve_date": "2026-09-21", "tenor_years": 0.25, "nominal_rate": 4.20, "nominal_rate_decimal": 0.042},
            {"curve_date": "2026-09-21", "tenor_years": 10.0, "nominal_rate": 4.80, "nominal_rate_decimal": 0.048},
        ],
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )

    path = tmp_path / "tdcsim_frn_rate_path.csv"
    write_forecast_rows_csv(path, rows)
    loaded = load_frn_rate_path(path, actuals_available_as_of="2026-09-30")

    assert loaded.iloc[0]["benchmark_rate_decimal"] == pytest.approx(0.042)
    assert loaded.iloc[0]["auction_high_rate_decimal"] == pytest.approx(0.042)
    assert loaded.iloc[0]["day_count_basis"] == pytest.approx(360.0)
    assert loaded.iloc[0]["lockout_business_days"] == pytest.approx(2.0)
    assert loaded.iloc[0]["source_role"] == "scenario_assumption"
    assert "future_frn_resets" in loaded.iloc[0]["claim_boundary"]


def test_build_forward_frn_rate_path_declares_lockout_for_daily_rows(tmp_path: Path) -> None:
    rows = build_forward_frn_rate_path_rows(
        scenario_id="baseline",
        periods=[
            SimpleNamespace(period_start=pd.Timestamp("2026-07-01").date(), period_end=pd.Timestamp("2026-07-02").date()),
            SimpleNamespace(period_start=pd.Timestamp("2026-07-02").date(), period_end=pd.Timestamp("2026-07-03").date()),
            SimpleNamespace(period_start=pd.Timestamp("2026-07-03").date(), period_end=pd.Timestamp("2026-07-04").date()),
        ],
        yield_surface_rows=[
            {"curve_date": "2026-07-02", "tenor_years": 0.25, "nominal_rate_decimal": 0.032},
            {"curve_date": "2026-07-03", "tenor_years": 0.25, "nominal_rate_decimal": 0.033},
            {"curve_date": "2026-07-04", "tenor_years": 0.25, "nominal_rate_decimal": 0.034},
        ],
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )

    path = tmp_path / "tdcsim_frn_rate_path_daily.csv"
    write_forecast_rows_csv(path, rows)
    loaded = load_frn_rate_path(path, actuals_available_as_of="2026-07-04")

    assert loaded["period_end"].tolist() == ["2026-07-02", "2026-07-03", "2026-07-04"]
    assert loaded["benchmark_rate_decimal"].tolist() == pytest.approx([0.032, 0.033, 0.034])
    assert loaded["lockout_business_days"].tolist() == pytest.approx([2.0, 2.0, 2.0])
    assert loaded["day_count_basis"].tolist() == pytest.approx([360.0, 360.0, 360.0])
