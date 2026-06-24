import pytest

from forecast_bundle_builders import build_debt_stock_path_rows, calculate_public_debt_bridge_components
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_debt_stock_path
from simulation_calendar import build_simulation_calendar


def test_debt_bridge_interpolates_public_debt_and_holds_bridge_components(tmp_path):
    periods = build_simulation_calendar("2026-06-30", "2027-09-30", "monthly")
    rows = build_debt_stock_path_rows(
        scenario_id="baseline",
        periods=periods,
        opening_state_date="2026-06-30",
        opening_cbo_federal_debt_held_public_bil=31000.0,
        cbo_fy_end_public_debt_targets_bil={2026: 32095.165, 2027: 33500.0},
        public_nonmarketable_treasury_bil=500.0,
        non_treasury_and_definition_residual_bil=25.0,
        observation_date="2026-06-30",
        available_date="2026-07-01",
    )

    by_end = {row["period_end"]: row for row in rows}
    expected_july = 31000.0 + (32095.165 - 31000.0) * (31 / 92)
    assert by_end["2026-07-31"]["cbo_federal_debt_held_public_target_bil"] == pytest.approx(expected_july)
    assert by_end["2026-07-31"]["anchor_type"] == "interpolated_assumption"
    assert by_end["2026-07-31"]["source_role"] == "scenario_assumption"

    assert by_end["2026-09-30"]["cbo_federal_debt_held_public_target_bil"] == pytest.approx(32095.165)
    assert by_end["2026-09-30"]["anchor_type"] == "cbo_fiscal_year_end"
    assert by_end["2026-09-30"]["source_role"] == "official_hard_anchor"
    assert by_end["2027-09-30"]["cbo_federal_debt_held_public_target_bil"] == pytest.approx(33500.0)

    for row in rows:
        assert row["public_nonmarketable_treasury_bil"] == pytest.approx(500.0)
        assert row["non_treasury_and_definition_residual_bil"] == pytest.approx(25.0)
        assert row["marketable_treasury_public_target_bil"] == pytest.approx(
            row["cbo_federal_debt_held_public_target_bil"] - 525.0
        )
        assert "issuance" not in row

    path = tmp_path / "tdcsim_debt_stock_path.csv"
    write_forecast_rows_csv(path, rows)
    loaded = load_debt_stock_path(path)
    assert len(loaded) == len(rows)
    assert set(loaded["interpolation_method"]) == {"linear_actual_days"}


def test_public_debt_bridge_uses_debt_to_penny_and_mspd_components():
    debt_to_penny_row = {
        "record_date": "2025-09-30",
        "debt_held_public_amt": "30277766440770.58",
    }
    mspd_rows = [
        {
            "record_date": "2025-09-30",
            "security_type_desc": "Total Nonmarketable",
            "debt_held_public_mil_amt": "582764.03699465",
        },
        {
            "record_date": "2025-09-30",
            "security_type_desc": "Total Public Debt Outstanding",
            "debt_held_public_mil_amt": "30277766.4202628",
        },
    ]

    bridge = calculate_public_debt_bridge_components(
        cbo_actual_federal_debt_held_public_bil=30172.402,
        debt_to_penny_row=debt_to_penny_row,
        mspd_rows=mspd_rows,
    )

    assert bridge["treasury_public_debt_bil"] == pytest.approx(30277.76644077058)
    assert bridge["public_nonmarketable_treasury_bil"] == pytest.approx(582.76403699465)
    assert bridge["non_treasury_and_definition_residual_bil"] == pytest.approx(-105.36444077058)
    assert bridge["bridge_method"] == "latest_actual_constant_nominal_by_component"


def test_public_debt_bridge_rejects_missing_debt_to_penny_reconciliation():
    with pytest.raises(ValueError, match="Debt to the Penny"):
        calculate_public_debt_bridge_components(
            cbo_actual_federal_debt_held_public_bil=30172.402,
            debt_to_penny_row={"record_date": "2025-09-30", "debt_held_public_amt": "30277766440770.58"},
            mspd_rows=[
                {
                    "record_date": "2025-09-30",
                    "security_type_desc": "Total Nonmarketable",
                    "debt_held_public_mil_amt": "582764.03699465",
                },
                {
                    "record_date": "2025-09-30",
                    "security_type_desc": "Total Public Debt Outstanding",
                    "debt_held_public_mil_amt": "30000000",
                },
            ],
        )
