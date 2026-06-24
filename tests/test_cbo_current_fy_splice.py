import pytest

from forecast_bundle_builders import (
    MTS_TABLE_1_DEFICIT_SELECTOR,
    MTS_TABLE_9_NET_INTEREST_SELECTOR,
    build_current_fy_splice_row,
    build_current_fy_splice_row_from_fiscaldata,
)
from forecast_input_builder import write_forecast_rows_csv
from forecast_paths import load_current_fy_splice


def test_current_fy_splice_computes_remaining_primary_deficit(tmp_path):
    row = build_current_fy_splice_row(
        scenario_id="baseline",
        fiscal_year=2026,
        simulation_start_date="2026-06-15",
        fiscal_actuals_through="2026-05-31",
        actuals_available_as_of="2026-06-10",
        cbo_full_fy_primary_deficit_bil=813.727,
        actual_total_deficit_fytd_bil=500.0,
        actual_net_interest_fytd_bil=310.0,
        mts_table_1_selector=MTS_TABLE_1_DEFICIT_SELECTOR,
        mts_table_9_selector=MTS_TABLE_9_NET_INTEREST_SELECTOR,
        observation_date="2026-05-31",
        available_date="2026-06-10",
    )

    assert row["actual_primary_deficit_fytd_bil"] == pytest.approx(190.0)
    assert row["remaining_cbo_primary_deficit_bil"] == pytest.approx(623.727)

    path = tmp_path / "tdcsim_current_fy_splice.csv"
    write_forecast_rows_csv(path, [row])
    loaded = load_current_fy_splice(path, actuals_available_as_of="2026-06-10")
    assert loaded.loc[0, "remaining_cbo_primary_deficit_bil"] == pytest.approx(623.727)


def test_current_fy_splice_rejects_future_actuals_without_lookahead(tmp_path):
    with pytest.raises(ValueError, match="available_date"):
        build_current_fy_splice_row(
            scenario_id="baseline",
            fiscal_year=2026,
            simulation_start_date="2026-06-15",
            fiscal_actuals_through="2026-05-31",
            actuals_available_as_of="2026-06-10",
            cbo_full_fy_primary_deficit_bil=813.727,
            actual_total_deficit_fytd_bil=500.0,
            actual_net_interest_fytd_bil=310.0,
            mts_table_1_selector=MTS_TABLE_1_DEFICIT_SELECTOR,
            mts_table_9_selector=MTS_TABLE_9_NET_INTEREST_SELECTOR,
            observation_date="2026-05-31",
            available_date="2026-06-11",
        )

    row = build_current_fy_splice_row(
        scenario_id="diagnostic",
        fiscal_year=2026,
        simulation_start_date="2026-06-15",
        fiscal_actuals_through="2026-05-31",
        actuals_available_as_of="2026-06-10",
        cbo_full_fy_primary_deficit_bil=813.727,
        actual_total_deficit_fytd_bil=500.0,
        actual_net_interest_fytd_bil=310.0,
        mts_table_1_selector=MTS_TABLE_1_DEFICIT_SELECTOR,
        mts_table_9_selector=MTS_TABLE_9_NET_INTEREST_SELECTOR,
        observation_date="2026-05-31",
        available_date="2026-06-11",
        allow_lookahead=True,
    )
    path = tmp_path / "lookahead_current_fy_splice.csv"
    write_forecast_rows_csv(path, [row])

    with pytest.raises(ValueError, match="available_date after actuals_available_as_of"):
        load_current_fy_splice(path, actuals_available_as_of="2026-06-10")
    loaded = load_current_fy_splice(path, actuals_available_as_of="2026-06-10", allow_lookahead=True)
    assert loaded.loc[0, "scenario_id"] == "diagnostic"


def test_current_fy_splice_requires_mts_table_9_net_interest_not_gross_interest_expense():
    table_1_deficit = {
        "record_date": "2026-05-31",
        "classification_desc": "Year-to-Date",
        "line_code_nbr": "280",
        "current_month_dfct_sur_amt": "1246203266386.93",
    }
    table_9_net_interest = {
        "record_date": "2026-05-31",
        "classification_desc": "Net Interest",
        "current_fytd_rcpt_outly_amt": "722706511243.20",
    }

    row = build_current_fy_splice_row_from_fiscaldata(
        scenario_id="baseline",
        fiscal_year=2026,
        simulation_start_date="2026-06-21",
        fiscal_actuals_through="2026-05-31",
        actuals_available_as_of="2026-06-16",
        cbo_full_fy_primary_deficit_bil=813.727,
        mts_table_1_row=table_1_deficit,
        mts_table_9_row=table_9_net_interest,
    )

    assert row["actual_net_interest_fytd_bil"] == pytest.approx(722.7065112432)
    assert row["actual_primary_deficit_fytd_bil"] == pytest.approx(523.49675514373)
    assert row["remaining_cbo_primary_deficit_bil"] == pytest.approx(290.23024485627)
    assert row["mts_table_1_selector"] == MTS_TABLE_1_DEFICIT_SELECTOR
    assert row["mts_table_9_selector"] == "classification_desc=Net Interest"

    gross_interest_expense_substitute = {
        "record_date": "2026-05-31",
        "classification_desc": "Government Account Series",
        "current_fytd_rcpt_outly_amt": "168566800000.00",
    }
    with pytest.raises(ValueError, match="Net Interest"):
        build_current_fy_splice_row_from_fiscaldata(
            scenario_id="baseline",
            fiscal_year=2026,
            simulation_start_date="2026-06-21",
            fiscal_actuals_through="2026-05-31",
            actuals_available_as_of="2026-06-16",
            cbo_full_fy_primary_deficit_bil=813.727,
            mts_table_1_row=table_1_deficit,
            mts_table_9_row=gross_interest_expense_substitute,
        )


def test_current_fy_splice_rejects_old_mts_table_1_federal_surplus_selector():
    old_table_1_deficit = {
        "record_date": "2026-05-31",
        "classification_desc": "Federal Surplus or Deficit",
        "line_code_nbr": "280",
        "current_fytd_dfct_sur_amt": "1246203266386.93",
    }
    table_9_net_interest = {
        "record_date": "2026-05-31",
        "classification_desc": "Net Interest",
        "current_fytd_rcpt_outly_amt": "722706511243.20",
    }

    with pytest.raises(ValueError, match="Year-to-Date line_code_nbr=280 using current_month_dfct_sur_amt"):
        build_current_fy_splice_row_from_fiscaldata(
            scenario_id="baseline",
            fiscal_year=2026,
            simulation_start_date="2026-06-21",
            fiscal_actuals_through="2026-05-31",
            actuals_available_as_of="2026-06-16",
            cbo_full_fy_primary_deficit_bil=813.727,
            mts_table_1_row=old_table_1_deficit,
            mts_table_9_row=table_9_net_interest,
        )
