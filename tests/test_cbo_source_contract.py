"""CBO February 2026 source-contract parser tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from forecast_input_builder import (
    BUDGET_WORKBOOK_SHA256,
    ECONOMIC_WORKBOOK_SHA256,
    build_cbo_fiscal_baseline_rows,
    build_cbo_macro_forecast_path_rows,
    parse_cbo_budget_source_contract,
    parse_cbo_economic_quarterly_source_contract,
    parse_cbo_grouped_table_2_1_exact_annual_data,
    sha256_file,
    validate_cbo_debt_continuity,
    validate_cbo_deficit_identity,
    verify_cbo_workbook_hashes,
    write_forecast_rows_csv,
)
from forecast_paths import load_cbo_fiscal_baseline, load_macro_forecast_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUDGET_WORKBOOK = PROJECT_ROOT / "ratewall/data/raw/cbo/51118-2026-02-Budget-Projections.xlsx"
ECONOMIC_WORKBOOK = PROJECT_ROOT / "ratewall/data/raw/cbo/51135-2026-02-Economic-Projections.xlsx"

pytestmark = pytest.mark.skipif(
    not BUDGET_WORKBOOK.exists() or not ECONOMIC_WORKBOOK.exists(),
    reason="optional local CBO workbook fixtures are not present",
)


def test_cbo_workbook_hashes_match_source_contract():
    assert sha256_file(BUDGET_WORKBOOK) == BUDGET_WORKBOOK_SHA256
    assert sha256_file(ECONOMIC_WORKBOOK) == ECONOMIC_WORKBOOK_SHA256
    assert verify_cbo_workbook_hashes(BUDGET_WORKBOOK, ECONOMIC_WORKBOOK) == {
        "budget": BUDGET_WORKBOOK_SHA256,
        "economic": ECONOMIC_WORKBOOK_SHA256,
    }


def test_cbo_budget_fy2026_fixture_values_and_identity():
    parsed = parse_cbo_budget_source_contract(BUDGET_WORKBOOK)

    assert parsed.value("cbo_total_deficit_bil", 2026) == pytest.approx(1852.703)
    assert parsed.value("primary_deficit_bil", 2026) == pytest.approx(813.727)
    assert parsed.value("cbo_net_interest_bil", 2026) == pytest.approx(1038.976)
    assert parsed.value("debt_held_public_end_bil", 2026) == pytest.approx(32095.165)

    residuals = validate_cbo_deficit_identity(parsed)
    fy2026 = [row for row in residuals if row.fiscal_year == 2026][0]
    assert fy2026.residual_bil == pytest.approx(0.0, abs=0.001)

    continuity = validate_cbo_debt_continuity(parsed)
    debt_fy2026 = [row for row in continuity if row.fiscal_year == 2026][0]
    assert debt_fy2026.residual_bil == pytest.approx(0.0, abs=0.001)


def test_cbo_primary_deficit_is_direct_raw_negative_row_not_reconstruction():
    parsed = parse_cbo_budget_source_contract(BUDGET_WORKBOOK)
    fixture = parsed.fixture("primary_deficit_bil", 2026)

    assert fixture.source_sheet == "Table 1-1"
    assert fixture.source_row_number == 30
    assert fixture.source_row_selector == "Primary deficit (-)"
    assert fixture.source_unit_block == "billions_of_dollars"
    assert fixture.source_year_or_period == "2026"
    assert fixture.raw_value == pytest.approx(-813.727)
    assert fixture.raw_sign_convention == "negative_deficit"
    assert fixture.canonical_transform == "negate_raw_value"
    assert fixture.canonical_value == pytest.approx(813.727)


def test_cbo_total_deficit_raw_canonical_sign_transform():
    parsed = parse_cbo_budget_source_contract(BUDGET_WORKBOOK)
    fixture = parsed.fixture("cbo_total_deficit_bil", 2026)

    assert fixture.source_sheet == "Table 1-1"
    assert fixture.source_row_number == 27
    assert fixture.source_row_selector == "Total deficit (-)"
    assert fixture.raw_value == pytest.approx(-1852.703)
    assert fixture.canonical_transform == "negate_raw_value"
    assert fixture.canonical_value == pytest.approx(1852.703)


def test_cbo_budget_table_1_3_selectors_and_units():
    parsed = parse_cbo_budget_source_contract(BUDGET_WORKBOOK)

    begin = parsed.fixture("debt_held_public_begin_bil", 2026)
    assert begin.source_sheet == "Table 1-3"
    assert begin.source_row_number == 9
    assert begin.source_row_selector == "Debt held by the public at the beginning of the year"
    assert begin.source_unit_block == "billions_of_dollars"
    assert begin.canonical_value == pytest.approx(30172.402)

    other = parsed.fixture("cbo_other_means_financing_bil", 2026)
    assert other.source_row_number == 13
    assert other.raw_sign_convention == "signed_source_value"
    assert other.canonical_value == pytest.approx(70.06)

    debt_identity = parsed.fixture("debt_identity_end_bil", 2026)
    assert debt_identity.source_row_number == 17
    assert debt_identity.source_row_selector == "Debt held by the public at the end of the year"
    assert debt_identity.source_unit_block == "billions_of_dollars"
    assert debt_identity.canonical_value == pytest.approx(32095.165)

    average_rate = parsed.fixture("cbo_average_interest_rate_pct", 2026)
    assert average_rate.source_row_number == 37
    assert average_rate.source_unit_block == "percent"
    assert average_rate.canonical_value == pytest.approx(3.404)


def test_cbo_quarterly_cpi_selector_row_53():
    parsed = parse_cbo_economic_quarterly_source_contract(ECONOMIC_WORKBOOK)
    fixture = parsed.fixture("cbo_cpi_u_index", "2026Q1")

    assert fixture.source_sheet == "1. Quarterly"
    assert fixture.source_row_number == 53
    assert fixture.source_row_selector == "Consumer price index, all urban consumers (CPI-U)"
    assert fixture.source_unit_block == "1982-84=100"
    assert fixture.canonical_value == pytest.approx(328.413)


def test_cbo_quarterly_10y_selector_row_103():
    parsed = parse_cbo_economic_quarterly_source_contract(ECONOMIC_WORKBOOK)
    fixture = parsed.fixture("cbo_10y_treasury_rate_pct", "2026Q1")

    assert fixture.source_sheet == "1. Quarterly"
    assert fixture.source_row_number == 103
    assert fixture.source_row_selector == "10-Year Treasury note"
    assert fixture.source_unit_block == "Percent"
    assert fixture.canonical_value == pytest.approx(4.059)


def test_cbo_quarterly_3m_selector_row_104():
    parsed = parse_cbo_economic_quarterly_source_contract(ECONOMIC_WORKBOOK)
    fixture = parsed.fixture("cbo_3m_tbill_rate_pct", "2026Q1")

    assert fixture.source_sheet == "1. Quarterly"
    assert fixture.source_row_number == 104
    assert fixture.source_row_selector == "3-Month Treasury bill"
    assert fixture.source_unit_block == "Percent"
    assert fixture.canonical_value == pytest.approx(3.469)


def test_cbo_fiscal_baseline_rows_validate_against_generic_loader(tmp_path: Path):
    parsed = parse_cbo_budget_source_contract(BUDGET_WORKBOOK)
    rows = build_cbo_fiscal_baseline_rows(parsed, scenario_id="central")
    fy2026 = [row for row in rows if row["fiscal_year"] == 2026][0]

    assert fy2026["primary_deficit_bil"] == pytest.approx(813.727)
    assert fy2026["source_role"] == "hard_input"
    assert fy2026["runtime_role"] == "hard_flow"
    assert "deficit_identity_residual_bil=0.000000000000" in fy2026["source_status"]
    assert "debt_continuity_residual_bil=0.000000000000" in fy2026["source_status"]

    output = tmp_path / "tdcsim_cbo_fiscal_baseline.csv"
    write_forecast_rows_csv(output, rows)
    loaded = load_cbo_fiscal_baseline(output, actuals_available_as_of="2026-02-11")
    assert loaded.loc[loaded["fiscal_year"] == 2026, "cbo_total_deficit_bil"].iloc[0] == pytest.approx(
        1852.703
    )


def test_cbo_macro_forecast_rows_validate_against_generic_loader(tmp_path: Path):
    parsed = parse_cbo_economic_quarterly_source_contract(ECONOMIC_WORKBOOK)
    rows = build_cbo_macro_forecast_path_rows(parsed, scenario_id="central")
    q1_2026 = [row for row in rows if row["period_start"] == "2026-01-01"][0]

    assert q1_2026["period_end"] == "2026-03-31"
    assert q1_2026["cbo_cpi_u_index"] == pytest.approx(328.413)
    assert q1_2026["cbo_3m_tbill_rate_pct"] == pytest.approx(3.469)
    assert q1_2026["cbo_10y_treasury_rate_pct"] == pytest.approx(4.059)
    assert q1_2026["cbo_cpi_u_inflation_pct"] == pytest.approx(0.7605834310)

    output = tmp_path / "tdcsim_macro_forecast_path.csv"
    write_forecast_rows_csv(output, rows)
    loaded = load_macro_forecast_path(output, actuals_available_as_of="2026-02-11")
    assert loaded.loc[loaded["period_start"] == "2026-01-01", "cbo_10y_treasury_rate_pct"].iloc[
        0
    ] == pytest.approx(4.059)


def test_cbo_grouped_table_2_1_rejected_as_exact_annual_data():
    with pytest.raises(ValueError, match="Grouped Table 2-1 values must not be used"):
        parse_cbo_grouped_table_2_1_exact_annual_data(ECONOMIC_WORKBOOK)
