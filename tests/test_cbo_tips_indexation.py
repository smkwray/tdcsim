from pathlib import Path

import pandas as pd
import pytest

from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_tips_cpi_path, load_tips_real_yield_path
from sim_pricing import calculate_tips_issue_price_ratio
from tips_indexation import (
    build_monthly_tips_cpi_path_rows,
    build_projected_cpi_lookup_from_macro,
    build_tips_daily_cpi_lookups_from_monthly_path,
    build_tips_real_yield_path_rows,
    enrich_opening_tips_from_cpi_detail,
)


def test_enrich_opening_tips_reconstructs_original_par_from_source_index_ratio(tmp_path: Path) -> None:
    cpi_detail = tmp_path / "tips_cpi.csv"
    pd.DataFrame(
        [
            {
                "cusip": "912TEST00",
                "original_issue_date": "2020-01-15",
                "index_date": "2026-06-21",
                "ref_cpi": 330.0,
                "index_ratio": 1.10,
                "pdf_link": "",
                "xml_link": "",
            }
        ]
    ).to_csv(cpi_detail, index=False)
    portfolio = pd.DataFrame(
        [
            {
                "BondID": 1,
                "SecurityType": "TIPS",
                "HolderType": "Private",
                "HolderSubBucket": "DomesticNonbank",
                "MaturityDate": pd.Timestamp("2030-01-15"),
                "FaceValue": 110.0,
                "OriginalPrincipal": 110.0,
                "AdjustedPrincipal": 110.0,
                "ReferenceCPI_Issue": 100.0,
                "IndexRatio": 1.0,
                "_SourceCUSIP": "912TEST00",
            }
        ]
    )

    enriched, metadata, diagnostics = enrich_opening_tips_from_cpi_detail(
        portfolio,
        cpi_detail_path=cpi_detail,
        index_date="2026-06-21",
    )

    row = enriched.iloc[0]
    assert row["AdjustedPrincipal"] == pytest.approx(110.0)
    assert row["OriginalPrincipal"] == pytest.approx(100.0)
    assert row["FaceValue"] == pytest.approx(100.0)
    assert row["ReferenceCPI_Issue"] == pytest.approx(300.0)
    assert row["IndexRatio"] == pytest.approx(1.10)
    assert metadata["opening_tips_reference_cpi"] == pytest.approx(330.0)
    assert metadata["opening_tips_indexation_accretion_embedded_bil"] == pytest.approx(10.0)
    assert diagnostics[0]["source_status"] == "treasurydirect_tips_cpi_detail_index_date_join"


def test_enrich_opening_tips_requires_complete_cusip_coverage(tmp_path: Path) -> None:
    cpi_detail = tmp_path / "tips_cpi.csv"
    pd.DataFrame(
        [
            {
                "cusip": "912OTHER0",
                "original_issue_date": "2020-01-15",
                "index_date": "2026-06-21",
                "ref_cpi": 330.0,
                "index_ratio": 1.10,
            }
        ]
    ).to_csv(cpi_detail, index=False)
    portfolio = pd.DataFrame(
        [
            {
                "BondID": 1,
                "SecurityType": "TIPS",
                "HolderType": "Private",
                "HolderSubBucket": "",
                "MaturityDate": pd.Timestamp("2030-01-15"),
                "AdjustedPrincipal": 110.0,
                "_SourceCUSIP": "912TEST00",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing 1 opening TIPS CUSIPs"):
        enrich_opening_tips_from_cpi_detail(
            portfolio,
            cpi_detail_path=cpi_detail,
            index_date="2026-06-21",
        )


def test_projected_reference_cpi_lookup_is_lagged_interpolated_and_anchored() -> None:
    macro = pd.DataFrame(
        [
            {
                "scenario_id": "baseline",
                "period_start": "2026-01-01",
                "period_end": "2026-03-31",
                "cbo_cpi_u_index": 100.0,
            },
            {
                "scenario_id": "baseline",
                "period_start": "2026-04-01",
                "period_end": "2026-06-30",
                "cbo_cpi_u_index": 110.0,
            },
            {
                "scenario_id": "baseline",
                "period_start": "2026-07-01",
                "period_end": "2026-09-30",
                "cbo_cpi_u_index": 130.0,
            },
        ]
    )
    dates = pd.to_datetime(["2026-06-01", "2026-07-01"])

    lookup = build_projected_cpi_lookup_from_macro(
        macro,
        dates,
        scenario_id="baseline",
        default_value=100.0,
        lag_months=3,
        anchor_date="2026-06-01",
        anchor_value=330.0,
    )

    assert lookup[pd.Timestamp("2026-06-01")] == pytest.approx(330.0)
    assert lookup[pd.Timestamp("2026-07-01")] > lookup[pd.Timestamp("2026-06-01")]


def test_monthly_tips_cpi_path_builds_daily_reference_cpi_from_cbo_anchors(tmp_path: Path) -> None:
    macro_rows = [
        {
            "scenario_id": "baseline",
            "period_start": "2026-04-01",
            "period_end": "2026-06-30",
            "cbo_cpi_u_index": 320.0,
        },
        {
            "scenario_id": "baseline",
            "period_start": "2026-07-01",
            "period_end": "2026-09-30",
            "cbo_cpi_u_index": 326.0,
        },
    ]

    rows = build_monthly_tips_cpi_path_rows(
        scenario_id="baseline",
        macro_forecast_rows=macro_rows,
        simulation_start_date="2026-06-21",
        simulation_end_date="2026-09-30",
        opening_reference_cpi=332.08433,
        reference_lag_months=3,
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )
    path = tmp_path / "tdcsim_tips_cpi_path.csv"
    write_forecast_rows_csv(path, rows)
    loaded = load_tips_cpi_path(path, actuals_available_as_of="2026-06-30")
    cpi_lookup, ref_lookup = build_tips_daily_cpi_lookups_from_monthly_path(
        loaded,
        pd.to_datetime(["2026-06-21", "2026-07-15"]),
        scenario_id="baseline",
        default_value=100.0,
        reference_lag_months=3,
    )

    assert loaded.iloc[0]["month"].endswith("-01")
    assert ref_lookup[pd.Timestamp("2026-06-21")] == pytest.approx(332.08433)
    assert cpi_lookup[pd.Timestamp("2026-07-15")] > cpi_lookup[pd.Timestamp("2026-06-21")]


def test_late_horizon_tips_expected_inflation_uses_terminal_cpi_rule(tmp_path: Path) -> None:
    macro_rows = [
        {
            "scenario_id": "baseline",
            "period_start": "2036-07-01",
            "period_end": "2036-09-30",
            "cbo_cpi_u_index": 100.0,
        },
        {
            "scenario_id": "baseline",
            "period_start": "2036-10-01",
            "period_end": "2036-12-31",
            "cbo_cpi_u_index": 102.0,
        },
    ]
    terminal_growth = (102.0 / 100.0) ** 4.0 - 1.0
    cpi_rows = build_monthly_tips_cpi_path_rows(
        scenario_id="baseline",
        macro_forecast_rows=macro_rows,
        simulation_start_date="2036-10-01",
        simulation_end_date="2036-10-01",
        pricing_horizon_end_date="2046-10-01",
        opening_reference_cpi=102.0,
        reference_lag_months=0,
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )

    cpi_path = tmp_path / "tdcsim_tips_cpi_path.csv"
    write_forecast_rows_csv(cpi_path, cpi_rows)
    loaded_cpi = load_tips_cpi_path(cpi_path, actuals_available_as_of="2026-06-30")
    terminal_rows = loaded_cpi.loc[loaded_cpi["cpi_horizon_role"].eq("tdcsim_terminal")]
    assert not terminal_rows.empty
    assert terminal_rows.iloc[0]["month"] == "2036-11-01"
    assert terminal_rows.iloc[-1]["month"] == "2046-11-01"
    assert float(terminal_rows.iloc[0]["terminal_annualized_cpi_growth_decimal"]) == pytest.approx(
        terminal_growth
    )
    assert "tdcsim_terminal_extrapolation_beyond_cbo_horizon" in terminal_rows.iloc[0]["source_status"]
    assert "not_cbo_cpi_coverage" in terminal_rows.iloc[0]["claim_boundary"]

    real_rows = build_tips_real_yield_path_rows(
        scenario_id="baseline",
        yield_surface_rows=[
            {
                "scenario_id": "baseline",
                "curve_date": "2036-10-01",
                "tenor_years": 10.0,
                "nominal_rate_decimal": 0.05,
            }
        ],
        tips_cpi_path_rows=cpi_rows,
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )
    real_path = tmp_path / "tdcsim_tips_real_yield_path.csv"
    write_forecast_rows_csv(real_path, real_rows)
    loaded_real = load_tips_real_yield_path(real_path, actuals_available_as_of="2026-06-30")
    row = loaded_real.iloc[0]

    assert float(row["expected_inflation_decimal"]) == pytest.approx(terminal_growth)
    assert float(row["expected_inflation_decimal"]) > 0.05
    assert float(row["real_yield_decimal"]) == pytest.approx(0.05 - terminal_growth)
    assert row["expected_inflation_cpi_terminal_rule"] == (
        "tdcsim_terminal_extrapolation_beyond_cbo_horizon_final_cbo_cpi_segment_annualized_growth"
    )
    assert "tdcsim_terminal_cpi_extrapolation" in row["claim_boundary"]


def test_tips_expected_inflation_requires_cpi_through_pricing_horizon() -> None:
    cpi_rows = build_monthly_tips_cpi_path_rows(
        scenario_id="baseline",
        macro_forecast_rows=[
            {
                "scenario_id": "baseline",
                "period_start": "2036-07-01",
                "period_end": "2036-09-30",
                "cbo_cpi_u_index": 100.0,
            },
            {
                "scenario_id": "baseline",
                "period_start": "2036-10-01",
                "period_end": "2036-12-31",
                "cbo_cpi_u_index": 102.0,
            },
        ],
        simulation_start_date="2036-10-01",
        simulation_end_date="2036-10-01",
        opening_reference_cpi=102.0,
        reference_lag_months=0,
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )

    with pytest.raises(ValueError, match="ends before requested pricing horizon"):
        build_tips_real_yield_path_rows(
            scenario_id="baseline",
            yield_surface_rows=[
                {
                    "scenario_id": "baseline",
                    "curve_date": "2036-10-01",
                    "tenor_years": 10.0,
                    "nominal_rate_decimal": 0.05,
                }
            ],
            tips_cpi_path_rows=cpi_rows,
            observation_date="2026-02-11",
            available_date="2026-02-11",
        )


def test_tips_real_yield_path_and_issue_price_support_premiums(tmp_path: Path) -> None:
    cpi_rows = build_monthly_tips_cpi_path_rows(
        scenario_id="baseline",
        macro_forecast_rows=[
            {"scenario_id": "baseline", "period_start": "2026-01-01", "period_end": "2026-03-31", "cbo_cpi_u_index": 300.0},
            {"scenario_id": "baseline", "period_start": "2027-01-01", "period_end": "2027-03-31", "cbo_cpi_u_index": 309.0},
        ],
        simulation_start_date="2026-01-01",
        simulation_end_date="2027-03-31",
        opening_reference_cpi=300.0,
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )
    real_rows = build_tips_real_yield_path_rows(
        scenario_id="baseline",
        yield_surface_rows=[
            {"scenario_id": "baseline", "curve_date": "2026-01-01", "tenor_years": 1.0, "nominal_rate_decimal": 0.015}
        ],
        tips_cpi_path_rows=cpi_rows,
        observation_date="2026-02-11",
        available_date="2026-02-11",
    )
    path = tmp_path / "tdcsim_tips_real_yield_path.csv"
    write_forecast_rows_csv(path, real_rows)
    loaded = load_tips_real_yield_path(path, actuals_available_as_of="2026-06-30")
    real_yield = float(loaded.iloc[0]["real_yield_decimal"])
    coupon = float(loaded.iloc[0]["real_coupon_decimal"])
    price = calculate_tips_issue_price_ratio(1.0, coupon, real_yield)

    assert real_yield < 0.0
    assert coupon >= 0.00125
    assert price > 1.0
