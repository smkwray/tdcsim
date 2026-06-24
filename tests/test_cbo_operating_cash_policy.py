import pytest

from forecast_bundle_builders import (
    build_cash_reconciliation_residual_rows,
    build_operating_cash_path_rows,
    tga_closing_balance_bil_from_dts_row,
)
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_cash_reconciliation_residual, load_operating_cash_path
from simulation_calendar import build_simulation_calendar


def test_operating_cash_path_supports_tga_only_real_and_nominal_modes(tmp_path):
    periods = build_simulation_calendar("2026-06-30", "2026-09-30", "monthly")
    cpi = {"2026-07-31": 303.0, "2026-08-31": 306.0, "2026-09-30": 309.0}
    real_rows = build_operating_cash_path_rows(
        scenario_id="real",
        periods=periods,
        base_date="2026-06-30",
        base_balance_bil=800.0,
        base_inflation_index_level=300.0,
        inflation_index_by_period_end=cpi,
        inflation_scalar=1.0,
        observation_date="2026-06-30",
        available_date="2026-07-01",
    )
    nominal_rows = build_operating_cash_path_rows(
        scenario_id="nominal",
        periods=periods,
        base_date="2026-06-30",
        base_balance_bil=800.0,
        inflation_scalar=0.0,
        observation_date="2026-06-30",
        available_date="2026-07-01",
    )

    assert real_rows[0]["operating_cash_target_bil"] == pytest.approx(808.0)
    assert real_rows[-1]["operating_cash_target_bil"] == pytest.approx(824.0)
    assert {row["operating_cash_definition"] for row in real_rows + nominal_rows} == {"tga_only"}
    assert {row["reserve_settlement_component"] for row in real_rows + nominal_rows} == {"tga"}
    assert {row["operating_cash_target_bil"] for row in nominal_rows} == {800.0}

    for row in real_rows + nominal_rows:
        assert row["operating_cash_target_bil"] == pytest.approx(
            row["tga_target_bil"] + row["ttl_target_bil"] + row["other_operating_cash_target_bil"]
        )

    path = tmp_path / "tdcsim_operating_cash_path.csv"
    write_forecast_rows_csv(path, real_rows + nominal_rows)
    loaded = load_operating_cash_path(path)
    assert set(loaded["inflation_scalar"]) == {0.0, 1.0}


def test_operating_cash_path_has_no_tga_floor():
    periods = build_simulation_calendar("2026-06-30", "2026-07-31", "monthly")
    rows = build_operating_cash_path_rows(
        scenario_id="stress",
        periods=periods,
        base_date="2026-06-30",
        base_balance_bil=-5.0,
        inflation_scalar=0.0,
        observation_date="2026-06-30",
        available_date="2026-07-01",
    )

    assert rows[0]["tga_target_bil"] == pytest.approx(-5.0)
    assert rows[0]["operating_cash_target_bil"] == pytest.approx(-5.0)


def test_cash_reconciliation_residual_affects_operating_cash_only(tmp_path):
    periods = build_simulation_calendar("2026-06-30", "2026-09-30", "monthly")
    rows = build_cash_reconciliation_residual_rows(
        scenario_id="baseline",
        periods=periods,
        cash_reconciliation_residual_bil={"2026-07-31": 2.5, "2026-08-31": -1.0, "2026-09-30": 0.0},
    )

    for row in rows:
        assert row["component_type"] == "unmodeled_nonbudget_cash_flow"
        assert row["affects_operating_cash"] is True
        assert row["affects_primary_deficit"] is False
        assert row["affects_net_interest"] is False
        assert row["affects_total_deficit"] is False
        assert row["affects_debt_target"] is False
        assert row["affects_issuance_size"] is False
        assert row["affects_tdc_fiscal_flow"] is False

    path = tmp_path / "tdcsim_cash_reconciliation_residual.csv"
    write_forecast_rows_csv(path, rows)
    loaded = load_cash_reconciliation_residual(path)
    assert loaded["affects_operating_cash"].all()
    for column in (
        "affects_primary_deficit",
        "affects_net_interest",
        "affects_total_deficit",
        "affects_debt_target",
        "affects_issuance_size",
        "affects_tdc_fiscal_flow",
    ):
        assert not loaded[column].any()


def test_tga_opening_balance_comes_from_dts_closing_balance_row():
    row = {
        "record_date": "2026-06-16",
        "account_type": "Treasury General Account (TGA) Closing Balance",
        "open_today_bal": "981113",
    }

    assert tga_closing_balance_bil_from_dts_row(row) == pytest.approx(981.113)

    with pytest.raises(ValueError, match="Closing Balance"):
        tga_closing_balance_bil_from_dts_row(
            {
                "record_date": "2026-06-16",
                "account_type": "Treasury General Account (TGA) Opening Balance",
                "open_today_bal": "979791",
            }
        )
