from pathlib import Path

import pandas as pd
import pytest

from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_tips_cpi_path, load_tips_real_yield_path
from sim_pricing import calculate_coupon_security_issue_price_ratio
from tips_indexation import (
    build_monthly_tips_cpi_path_rows,
    build_projected_cpi_lookup_from_macro,
    build_tips_daily_cpi_lookups_from_monthly_path,
    build_tips_real_yield_path_rows,
    enrich_opening_tips_from_cpi_detail,
    opening_tips_to_par,
)


def test_enrich_opening_tips_values_original_par_at_the_close(tmp_path: Path) -> None:
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
                "FaceValue": 100.0,
                "OriginalPrincipal": 100.0,
                "AdjustedPrincipal": 109.0,
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
    price = calculate_coupon_security_issue_price_ratio(1.0, coupon, real_yield)

    assert real_yield < 0.0
    assert coupon >= 0.00125
    assert price > 1.0

def test_opening_par_uses_may_ratio_and_published_auction_cpi_for_new_cusip(tmp_path: Path) -> None:
    book = pd.DataFrame([
        {"SecurityType": "TIPS", "CUSIP": "SEASONED", "AdjustedPrincipal": 110.0, "FaceValue": 110.0},
        {"SecurityType": "TIPS", "CUSIP": "NEW", "AdjustedPrincipal": 101.502, "FaceValue": 101.502},
    ])
    detail = pd.DataFrame([
        {"index_date": pd.Timestamp("2026-05-29"), "cusip": "SEASONED", "index_ratio": 1.10, "ref_cpi": 329.88126},
        {"index_date": pd.Timestamp("2026-06-18"), "cusip": "NEW", "index_ratio": 1.01790, "ref_cpi": 331.80363},
    ])
    auctions = tmp_path / "auctions.csv"
    pd.DataFrame([{"cusip": "NEW", "auction_date": "2026-04-23", "ref_cpi_on_dated_date": 325.0}]).to_csv(auctions, index=False)
    par, sources = opening_tips_to_par(
        book, detail=detail, auctions_path=auctions, valuation_date="2026-05-29", cusip_column="CUSIP",
    )
    assert par["FaceValue"].tolist() == pytest.approx([100, 100], abs=1e-6)
    assert sources["NEW"] == "treasury_daily_reference_cpi_divided_by_auction_dated_reference_cpi"


def test_opening_tips_verifier_uses_independent_mspd_tranche_par(tmp_path: Path) -> None:
    from scripts.verify_cbo_forecast_package import _verify_opening_tips_against_mspd

    fiscaldata = tmp_path / "sources" / "fiscaldata"
    opening = tmp_path / "sources" / "opening"
    fiscaldata.mkdir(parents=True)
    opening.mkdir()
    # 1000 issued + 100 reopened - 100 redeemed leaves 1000 par. USD 1100
    # million reported adjusted principal scales to the public cohort's 0.88
    # billion. The published June ratio then marks 0.8 par to 0.904 billion.
    fields = {
        "record_date": "2026-05-31", "security_type_desc": "Marketable",
        "security_class1_desc": "Inflation-Protected Securities", "security_class2_desc": "TIPS-A",
    }
    pd.DataFrame([
        {**fields, "issued_amt": 1000, "redeemed_amt": 0, "outstanding_amt": None},
        {**fields, "issued_amt": 100, "redeemed_amt": -100, "outstanding_amt": 1100},
    ]).to_csv(fiscaldata / "mspd_table_3_market_2026-05-31.csv", index=False)
    pd.DataFrame([{
        "SecurityType": "TIPS", "cusip_or_synthetic_id": "TIPS-A", "AdjustedPrincipal": 0.88,
    }]).to_csv(opening / "tdcsim_initial_portfolio_cohorts.csv", index=False)
    pd.DataFrame([{
        "cusip": "TIPS-A", "index_date": "2026-06-18", "index_ratio": 1.13,
    }]).to_csv(fiscaldata / "tips_cpi_data_detail.csv", index=False)
    lot = {"IssueID": "TIPS-A", "OriginalPrincipal": "0.8", "FaceValue": "0.8", "AdjustedPrincipal": "0.904"}
    _verify_opening_tips_against_mspd(tmp_path, [lot])
    with pytest.raises(ValueError, match="MSPD public par"):
        _verify_opening_tips_against_mspd(tmp_path, [{**lot, "OriginalPrincipal": "0.79", "FaceValue": "0.79"}])
    with pytest.raises(ValueError, match="MSPD June 18 adjusted principal"):
        _verify_opening_tips_against_mspd(tmp_path, [{**lot, "AdjustedPrincipal": "0.9"}])


def test_tips_path_carries_observed_reference_cpi_through_opening_weekend() -> None:
    macro = [
        {"scenario_id": "base", "period_start": "2026-04-01", "cbo_cpi_u_index": 331.0},
        {"scenario_id": "base", "period_start": "2026-07-01", "cbo_cpi_u_index": 334.0},
    ]
    rows = build_monthly_tips_cpi_path_rows(
        scenario_id="base", macro_forecast_rows=macro,
        simulation_start_date="2026-06-21", simulation_end_date="2026-09-30",
        reference_cpi_anchor_date="2026-06-18", opening_reference_cpi=331.80363,
        observed_cpi_by_month={"2026-03-01": 330.213, "2026-04-01": 333.020},
        observation_date="2026-02-11", available_date="2026-02-11",
    )
    days = pd.to_datetime(["2026-06-18", "2026-06-21"])
    _, references = build_tips_daily_cpi_lookups_from_monthly_path(
        pd.DataFrame(rows), days, scenario_id="base", default_value=0.0,
    )
    assert rows[0]["anchor_date"] == "2026-06-18"
    assert references == dict(zip(days, (331.80363, 332.08433)))
