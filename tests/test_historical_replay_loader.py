"""Tests for historical replay CSV loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_loader import (
    enrich_mspd_cohorts_with_security_sources,
    load_auction_flows,
    load_fixed_coupon_auction_events,
    load_fixed_coupon_monthly_stocks,
    load_frn_auction_lots,
    load_frn_benchmark_auctions,
    load_frn_daily_index_validation,
    load_frn_mspd_principal_controls,
    load_interest_auction_lots,
    load_mspd_cohorts,
    load_nonbill_discount_premium_auction_lots,
    load_quarterly_cash,
    load_sector_positions,
    load_tips_auction_events,
    load_tips_cpi_reference_path,
    load_tips_inflation_adjustment_stocks,
)


def test_load_quarterly_cash_normalizes_aliases_and_filters_range(tmp_path):
    path = tmp_path / "cash.csv"
    pd.DataFrame(
        [
            {
                "quarter_end": "2024-03-31",
                "selected_operating_cash_level_mil": 500.0,
                "selected_operating_cash_level_source": "dts_tga_plus_ttl",
                "selected_operating_cash_tx_mil": -20.0,
                "selected_operating_cash_tx_source": "z1_quarterly_treasury_operating_cash_transaction",
                "dts_tga_mil": 450.0,
                "dts_ttl_mil": 50.0,
                "evidence_label": "observed",
            },
            {
                "quarter_end": "2024-06-30",
                "selected_operating_cash_level_mil": 520.0,
                "selected_operating_cash_level_source": "dts_tga_plus_ttl",
                "selected_operating_cash_tx_mil": 20.0,
                "selected_operating_cash_tx_source": "z1_quarterly_treasury_operating_cash_transaction",
                "dts_tga_mil": 500.0,
                "dts_ttl_mil": 20.0,
                "evidence_label": "observed",
            },
        ]
    ).to_csv(path, index=False)

    result = load_quarterly_cash(path, start_quarter="2024Q2")

    assert result["quarter"].tolist() == ["2024Q2"]
    assert result.loc[0, "toc"] == 520.0
    assert result.loc[0, "toc_transaction"] == 20.0
    assert result.loc[0, "toc_transaction_source_status"] == "z1_quarterly_treasury_operating_cash_transaction"
    assert result.loc[0, "tga_source_status"] == "observed"
    assert result.loc[0, "ttl_source_status"] == "observed"


def test_load_sector_positions_normalizes_columns(tmp_path):
    path = tmp_path / "sector.csv"
    pd.DataFrame(
        [
            {
                "date": "2024-03-31",
                "z1_sector": "banks",
                "broad_investor_class": "banks",
                "measure": "tx",
                "value": "15.25",
                "source_file": "fred_z1.csv",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(path, index=False)

    result = load_sector_positions(path)

    assert result.columns.tolist() == [
        "quarter",
        "sector",
        "native_sector",
        "broad_holder_class",
        "measure",
        "value",
        "source_status",
        "source_file",
        "evidence_label",
    ]
    assert result.loc[0, "quarter"] == "2024Q1"
    assert result.loc[0, "sector"] == "banks"
    assert result.loc[0, "native_sector"] == "banks"
    assert result.loc[0, "broad_holder_class"] == "banks"
    assert result.loc[0, "measure"] == "transaction"
    assert result.loc[0, "value"] == 15.25
    assert result.loc[0, "source_file"] == "fred_z1.csv"


def test_load_sector_positions_preserves_z1_series_basis_metadata(tmp_path):
    path = tmp_path / "sector_z1.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2024Q1",
                "z1_series": "LM263061105.Q",
                "z1_code": "263061105",
                "z1_sector": "rest_of_world",
                "broad_investor_class": "foreign_international",
                "measure": "level",
                "value": 100.0,
                "source_file": "z1_csv_files.zip::csv/l210.csv",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(path, index=False)

    result = load_sector_positions(path)

    assert result.loc[0, "z1_series"] == "LM263061105.Q"
    assert result.loc[0, "z1_code"] == "263061105"
    assert result.loc[0, "source_file"] == "z1_csv_files.zip::csv/l210.csv"


def test_load_sector_positions_aggregates_duplicate_keys(tmp_path):
    path = tmp_path / "sector_duplicate.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2024Q1",
                "sector": "banks",
                "measure": "level",
                "value": 10.0,
                "source_status": "fred_z1",
                "evidence_label": "observed",
            },
            {
                "quarter": "2024Q1",
                "sector": "banks",
                "measure": "level",
                "value": 11.0,
                "source_status": "fred_z1",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(path, index=False)

    result = load_sector_positions(path)

    assert len(result) == 1
    assert result.loc[0, "quarter"] == "2024Q1"
    assert result.loc[0, "sector"] == "banks"
    assert result.loc[0, "value"] == 21.0


def test_load_auction_flows_normalizes_schema(tmp_path):
    path = tmp_path / "auction.csv"
    pd.DataFrame(
        [
            {
                "auction_date": "2024-01-10",
                "issue_date": "2024-01-15",
                "maturity_date": "2025-01-15",
                "cusip": "912345AA1",
                "security_type": "Note",
                "offering_amt": 1000.0,
                "total_accepted": 980.0,
                "avg_med_yield": 4.2,
                "int_rate": 4.0,
                "price_per100": 99.5,
                "source_status": "fiscaldata_auction_query",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(path, index=False)

    result = load_auction_flows(path)

    assert result.loc[0, "quarter"] == "2024Q1"
    assert result.loc[0, "security_type"] == "note"
    assert result.loc[0, "auction_id"] == "912345AA1|2024-01-10"
    assert result.loc[0, "offering_amount"] == 1000.0
    assert result.loc[0, "coupon_rate_decimal"] == pytest.approx(0.04)
    assert result.loc[0, "issue_price_ratio"] == pytest.approx(0.995)
    assert result.loc[0, "issue_yield_decimal"] == pytest.approx(0.042)
    assert result.loc[0, "original_maturity_years"] == pytest.approx(1.0, abs=0.005)


def test_load_auction_flows_rejects_invalid_evidence_labels(tmp_path):
    path = tmp_path / "auction_bad_label.csv"
    pd.DataFrame(
        [
            {
                "auction_date": "2024-01-10",
                "issue_date": "2024-01-15",
                "maturity_date": "2025-01-15",
                "cusip": "912345AA1",
                "security_type": "Bill",
                "offering_amt": 1000.0,
                "source_status": "fiscaldata_auction_query",
                "evidence_label": "bad_label",
            }
        ]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="invalid evidence labels"):
        load_auction_flows(path)


def test_load_interest_auction_lots_filters_by_accrual_overlap(tmp_path):
    path = tmp_path / "auction_lots.csv"
    pd.DataFrame(
        [
            {
                "auction_date": "2024-12-26",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-03-31",
                "cusip": "912345AA1",
                "security_type": "Bill",
                "security_term": "13-Week",
                "avg_med_price": 99.0,
                "offering_amt": 100_000_000.0,
                "total_accepted": 101_000_000.0,
                "comp_accepted": 80_000_000.0,
                "noncomp_accepted": 1_000_000.0,
                "soma_accepted": 20_000_000.0,
            },
            {
                "auction_date": "2024-12-26",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-03-31",
                "cusip": "912345BB2",
                "security_type": "Note",
                "security_term": "2-Year",
                "avg_med_price": 100.0,
                "offering_amt": 100_000_000.0,
                "total_accepted": 100_000_000.0,
            },
            {
                "auction_date": "2023-12-26",
                "issue_date": "2023-12-31",
                "maturity_date": "2024-03-31",
                "cusip": "912345CC3",
                "security_type": "Bill",
                "security_term": "13-Week",
                "avg_med_price": 99.0,
                "offering_amt": 100_000_000.0,
                "total_accepted": 100_000_000.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_interest_auction_lots(path, start_quarter="2025Q1", end_quarter="2025Q1")

    assert len(result) == 1
    assert result.loc[0, "lot_id"] == "912345AA1|2024-12-26|2024-12-31"
    assert result.loc[0, "price_per100"] == pytest.approx(99.0)
    assert result.loc[0, "accepted_component_sum"] == pytest.approx(101_000_000.0)


def test_load_fixed_coupon_monthly_stocks_keeps_monthly_notes_bonds_and_sums_duplicates(tmp_path):
    path = tmp_path / "mspd_coupon.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-01-31",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Notes",
                "security_class2_desc": "912345AA1",
                "interest_rate_pct": 4.0,
                "maturity_date": "2029-01-15",
                "interest_pay_date_1": "01/15",
                "interest_pay_date_2": "07/15",
                "outstanding_amt": 1000.0,
            },
            {
                "record_date": "2024-01-31",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Notes",
                "security_class2_desc": "912345AA1",
                "interest_rate_pct": 4.0,
                "maturity_date": "2029-01-15",
                "interest_pay_date_1": "01/15",
                "interest_pay_date_2": "07/15",
                "outstanding_amt": 25.0,
            },
            {
                "record_date": "2024-01-31",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Treasury Bills",
                "security_class2_desc": "912345BB2",
                "interest_rate_pct": 0.0,
                "maturity_date": "2024-04-15",
                "interest_pay_date_1": "04/15",
                "interest_pay_date_2": "",
                "outstanding_amt": 1000.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_fixed_coupon_monthly_stocks(path, start_quarter="2024Q1", end_quarter="2024Q1")

    assert len(result) == 1
    assert result.loc[0, "record_date"] == "2024-01-31"
    assert result.loc[0, "cusip"] == "912345AA1"
    assert result.loc[0, "security_class"] == "Notes"
    assert result.loc[0, "outstanding_mil"] == pytest.approx(1025.0)
    assert result.loc[0, "coupon_rate_decimal"] == pytest.approx(0.04)


def test_load_fixed_coupon_monthly_stocks_returns_empty_for_quarterly_fixture_shape(tmp_path):
    path = tmp_path / "cohorts.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "BILL1",
                "security_type": "bill",
                "issue_date": "2025-01-01",
                "maturity_date": "2025-06-30",
                "outstanding": 100.0,
            }
        ]
    ).to_csv(path, index=False)

    result = load_fixed_coupon_monthly_stocks(path)

    assert result.empty
    assert "record_date" in result.columns
    assert "coupon_rate_decimal" in result.columns


def test_load_fixed_coupon_auction_events_filters_nominal_fixed_notes_and_bonds(tmp_path):
    path = tmp_path / "auctions_coupon.csv"
    pd.DataFrame(
        [
            {
                "auction_date": "2024-01-10",
                "issue_date": "2024-01-15",
                "maturity_date": "2029-01-15",
                "cusip": "912345AA1",
                "security_type": "Note",
                "floating_rate": "No",
                "inflation_index_security": "No",
                "total_accepted": 42_000_000_000.0,
                "int_rate": 4.0,
            },
            {
                "auction_date": "2024-01-10",
                "issue_date": "2024-01-15",
                "maturity_date": "2026-01-15",
                "cusip": "912345FRN",
                "security_type": "Note",
                "floating_rate": "Yes",
                "inflation_index_security": "No",
                "total_accepted": 42_000_000_000.0,
                "int_rate": 4.0,
            },
            {
                "auction_date": "2024-01-10",
                "issue_date": "2024-01-15",
                "maturity_date": "2034-01-15",
                "cusip": "912345TII",
                "security_type": "Note",
                "floating_rate": "No",
                "inflation_index_security": "Yes",
                "total_accepted": 42_000_000_000.0,
                "int_rate": 4.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_fixed_coupon_auction_events(path, start_quarter="2024Q1", end_quarter="2024Q1")

    assert len(result) == 1
    assert result.loc[0, "event_id"] == "912345AA1|2024-01-10|2024-01-15"
    assert result.loc[0, "security_class"] == "Notes"
    assert result.loc[0, "total_accepted_mil"] == pytest.approx(42_000.0)


def test_load_nonbill_discount_premium_auction_lots_preserves_price_tiers_and_frn_exclusion_inputs(tmp_path):
    path = tmp_path / "auctions_dp.csv"
    pd.DataFrame(
        [
            {
                "cusip": "912345AA1",
                "security_type": "Note",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "first_int_payment_date": "2025-07-15",
                "maturity_date": "2030-01-15",
                "floating_rate": "No",
                "inflation_index_security": "No",
                "reopening": "No",
                "auction_format": "Single-Price",
                "total_accepted": 1_000_000_000.0,
                "int_rate": 4.0,
                "price_per100": 99.5,
                "high_yield": 4.1,
            },
            {
                "cusip": "912345TI1",
                "security_type": "TIPS",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "first_int_payment_date": "2025-07-15",
                "maturity_date": "2030-01-15",
                "floating_rate": "No",
                "inflation_index_security": "Yes",
                "reopening": "Yes",
                "total_accepted": 2_000_000_000.0,
                "int_rate": 1.0,
                "unadj_price": 0.0,
                "adj_price": 105.0,
                "index_ratio_on_issue_date": 1.05,
                "high_yield": 1.1,
            },
            {
                "cusip": "912345FRN",
                "security_type": "Note",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "first_int_payment_date": "2025-07-15",
                "maturity_date": "2027-01-15",
                "floating_rate": "Yes",
                "inflation_index_security": "No",
                "total_accepted": 3_000_000_000.0,
                "int_rate": 0.0,
                "price_per100": 100.1,
                "spread": 0.2,
            },
            {
                "cusip": "912796AA1",
                "security_type": "Bill",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "maturity_date": "2025-04-15",
                "total_accepted": 4_000_000_000.0,
                "price_per100": 99.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_nonbill_discount_premium_auction_lots(path, start_quarter="2025Q1", end_quarter="2025Q1")

    assert set(result["instrument_family"]) == {
        "Treasury Notes",
        "Treasury Inflation Protected Securities (TIPS)",
        "Treasury Floating Rate Notes (FRN)",
    }
    tips = result.loc[result["cusip"].eq("912345TI1")].iloc[0]
    assert tips["clean_price_per100"] == pytest.approx(100.0)
    assert tips["price_source_tier"] == "adj_price_div_index_ratio"
    assert bool(tips["is_reopening"]) is True


def test_load_frn_benchmark_auctions_converts_13_week_discount_rate(tmp_path):
    path = tmp_path / "auctions.csv"
    pd.DataFrame(
        [
            {
                "auction_date": "2024-10-28",
                "issue_date": "2024-10-31",
                "maturity_date": "2025-01-30",
                "security_type": "Bill",
                "security_term": "13-Week",
                "high_discnt_rate": 4.51,
            },
            {
                "auction_date": "2024-10-28",
                "issue_date": "2024-10-31",
                "maturity_date": "2025-04-30",
                "security_type": "Bill",
                "security_term": "26-Week",
                "high_discnt_rate": 4.51,
            },
        ]
    ).to_csv(path, index=False)

    result = load_frn_benchmark_auctions(path)

    assert len(result) == 1
    assert result.loc[0, "effective_date"] == "2024-10-29"
    expected = 4.51 / (1.0 - (4.51 / 100.0) * 91.0 / 360.0)
    assert result.loc[0, "benchmark_index_pct"] == pytest.approx(round(expected, 9))


def test_load_frn_auction_lots_filters_floating_rate_rows(tmp_path):
    path = tmp_path / "frn_auction.csv"
    pd.DataFrame(
        [
            {
                "cusip": "912345FRN",
                "auction_date": "2024-10-29",
                "issue_date": "2024-10-31",
                "maturity_date": "2026-10-31",
                "dated_date": "2024-10-31",
                "original_dated_date": "2024-10-31",
                "security_type": "Note",
                "floating_rate": "Yes",
                "reopening": "No",
                "spread": 0.205,
                "total_accepted": 15_000_000_000.0,
                "frn_index_determination_date": "2024-10-21",
                "frn_index_determination_rate": 4.51,
            },
            {
                "cusip": "912345FIX",
                "auction_date": "2024-10-29",
                "issue_date": "2024-10-31",
                "maturity_date": "2026-10-31",
                "security_type": "Note",
                "floating_rate": "No",
                "reopening": "No",
                "spread": "",
                "total_accepted": 15_000_000_000.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_frn_auction_lots(path)

    assert len(result) == 1
    assert result.loc[0, "lot_id"] == "912345FRN|2024-10-29|2024-10-31"
    assert result.loc[0, "total_accepted_mil"] == pytest.approx(15_000.0)
    assert bool(result.loc[0, "is_reopening"]) is False


def test_load_frn_daily_index_validation_normalizes_source_rows(tmp_path):
    path = tmp_path / "frn_daily.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-10-31",
                "cusip": "912345FRN",
                "start_of_accrual_period": "2024-10-30",
                "end_of_accrual_period": "2024-10-31",
                "daily_index": 4.562008,
                "spread": 0.205,
                "daily_int_accrual_rate": 4.767008,
                "daily_accrued_int_per100": 0.0132416889,
            }
        ]
    ).to_csv(path, index=False)

    result = load_frn_daily_index_validation(path)

    assert result.loc[0, "quarter"] == "2024Q4"
    assert result.loc[0, "daily_index"] == pytest.approx(4.562008)


def test_load_frn_mspd_principal_controls_prefers_observed_outstanding(tmp_path):
    path = tmp_path / "mspd.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-10-31",
                "security_class1_desc": "Floating Rate Notes",
                "security_class2_desc": "912345FRN",
                "outstanding_amt": 50_000.0,
                "issued_amt": "",
                "redeemed_amt": "",
            },
            {
                "record_date": "2024-10-31",
                "security_class1_desc": "Floating Rate Notes",
                "security_class2_desc": "912345FRN",
                "outstanding_amt": "",
                "issued_amt": 2_000.0,
                "redeemed_amt": "",
            },
        ]
    ).to_csv(path, index=False)

    result = load_frn_mspd_principal_controls(path)

    assert len(result) == 1
    assert result.loc[0, "control_outstanding_mil"] == pytest.approx(50_000.0)
    assert result.loc[0, "mspd_row_count"] == 2


def test_load_frn_mspd_principal_controls_returns_empty_for_quarterly_fixture_shape(tmp_path):
    path = tmp_path / "cohorts.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "FRN1",
                "security_type": "frn",
                "issue_date": "2024-01-01",
                "maturity_date": "2026-01-31",
                "outstanding": 100.0,
            }
        ]
    ).to_csv(path, index=False)

    result = load_frn_mspd_principal_controls(path)

    assert result.empty
    assert "record_date" in result.columns
    assert "control_outstanding_mil" in result.columns


def test_load_mspd_cohorts_normalizes_required_columns(tmp_path):
    path = tmp_path / "mspd.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912345BB2",
                "issue_date": "2024-01-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": 2500.0,
                "issued_amt": 2600.0,
                "redeemed_amt": 100.0,
                "source_status": "mspd_table_3_market",
                "evidence_label": "observed",
            }
        ]
    ).to_csv(path, index=False)

    result = load_mspd_cohorts(path)

    assert result.loc[0, "quarter"] == "2024Q1"
    assert result.loc[0, "cohort_id"] == "912345BB2|2024-01-15|2024-07-15"
    assert result.loc[0, "cusip"] == "912345BB2"
    assert result.loc[0, "security_type"] == "bill"
    assert result.loc[0, "outstanding"] == 2500.0
    assert result.loc[0, "original_maturity_years"] == pytest.approx(0.5, abs=0.01)


def test_load_mspd_cohorts_snaps_raw_monthly_rows_to_quarter_end_and_aggregates(tmp_path):
    path = tmp_path / "mspd_raw.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-01-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912345BB2",
                "series_cd": "null",
                "issue_date": "2024-01-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": 1000.0,
                "issued_amt": 1000.0,
                "redeemed_amt": 0.0,
            },
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912345BB2",
                "series_cd": "null",
                "issue_date": "2024-01-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": 2400.0,
                "issued_amt": 2400.0,
                "redeemed_amt": 0.0,
            },
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912345BB2",
                "series_cd": "null",
                "issue_date": "2024-02-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": "*  ",
                "issued_amt": 100.0,
                "redeemed_amt": 0.0,
            },
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Marketable",
                "security_class2_desc": "Total Treasury Bills",
                "series_cd": "null",
                "issue_date": "2024-01-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": 2400.0,
                "issued_amt": 2400.0,
                "redeemed_amt": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_mspd_cohorts(path)

    assert len(result) == 1
    assert result.loc[0, "quarter"] == "2024Q1"
    assert result.loc[0, "cohort_id"] == "912345BB2|2024-01-15|2024-07-15"
    assert result.loc[0, "cusip"] == "912345BB2"
    assert result.loc[0, "outstanding"] == 2400.0


def test_load_mspd_cohorts_keeps_reopenings_as_distinct_tranches(tmp_path):
    path = tmp_path / "mspd_reopenings.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912345BB2",
                "series_cd": "A",
                "issue_date": "2024-01-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": 2400.0,
                "yield_pct": 5.0,
            },
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912345BB2",
                "series_cd": "B",
                "issue_date": "2024-02-15",
                "maturity_date": "2024-07-15",
                "outstanding_amt": 100.0,
                "yield_pct": 5.5,
            },
        ]
    ).to_csv(path, index=False)

    result = load_mspd_cohorts(path)

    assert result["cohort_id"].tolist() == [
        "912345BB2|2024-01-15|2024-07-15",
        "912345BB2|2024-02-15|2024-07-15",
    ]
    assert result["cusip"].tolist() == ["912345BB2", "912345BB2"]
    assert result["outstanding"].tolist() == pytest.approx([2400.0, 100.0])
    assert result["issue_yield_decimal"].tolist() == pytest.approx([0.05, 0.055])


def test_load_mspd_cohorts_decomposes_reopening_aggregate_outstanding(tmp_path):
    path = tmp_path / "mspd_reopening_aggregate.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2025-12-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912797RA7",
                "issue_date": "2025-07-03",
                "maturity_date": "2026-01-02",
                "outstanding_amt": 262107.6530,
                "issued_amt": 75772.2680,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
            },
            {
                "record_date": "2025-12-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912797RA7",
                "issue_date": "2025-10-02",
                "maturity_date": "2026-01-02",
                "outstanding_amt": "",
                "issued_amt": 86712.4442,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
            },
            {
                "record_date": "2025-12-31",
                "security_type_desc": "Bills Maturity Value",
                "security_class2_desc": "912797RA7",
                "issue_date": "2025-11-20",
                "maturity_date": "2026-01-02",
                "outstanding_amt": "",
                "issued_amt": 99622.9408,
                "redeemed_amt": 0.0,
                "inflation_adj_amt": 0.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_mspd_cohorts(path)

    assert len(result) == 3
    assert result["cohort_id"].tolist() == [
        "912797RA7|2025-07-03|2026-01-02",
        "912797RA7|2025-10-02|2026-01-02",
        "912797RA7|2025-11-20|2026-01-02",
    ]
    assert result["outstanding"].tolist() == pytest.approx([75772.2680, 86712.4442, 99622.9408])
    assert result["outstanding"].sum() == pytest.approx(262107.6530)
    assert result["source_status"].str.contains("mspd_reopening_outstanding_decomposed").all()


def test_load_mspd_cohorts_drops_positive_summary_rows_without_real_cusip(tmp_path):
    path = tmp_path / "mspd_summary.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Federal Financing Bank",
                "security_class2_desc": "",
                "series_cd": "AA",
                "issue_date": "",
                "maturity_date": "",
                "outstanding_amt": 4000.0,
            },
            {
                "record_date": "2024-03-31",
                "security_type_desc": "Marketable",
                "security_class1_desc": "Notes",
                "security_class2_desc": "912345CC3",
                "series_cd": "AA",
                "issue_date": "2024-01-15",
                "maturity_date": "2029-01-15",
                "outstanding_amt": 1000.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_mspd_cohorts(path)

    assert len(result) == 1
    assert result.loc[0, "cusip"] == "912345CC3"


def test_load_mspd_cohorts_aggregates_duplicate_quarter_cohort_keys(tmp_path):
    path = tmp_path / "mspd_duplicate.csv"
    pd.DataFrame(
        [
            {
                "quarter": "2024Q1",
                "cohort_id": "cohort-1",
                "security_type": "note",
                "issue_date": "2024-01-15",
                "maturity_date": "2025-01-15",
                "outstanding": 100.0,
                "source_status": "mspd_table_3_market",
                "evidence_label": "observed",
            },
            {
                "quarter": "2024Q1",
                "cohort_id": "cohort-1",
                "security_type": "note",
                "issue_date": "2024-01-15",
                "maturity_date": "2025-01-15",
                "outstanding": 120.0,
                "source_status": "mspd_table_3_market",
                "evidence_label": "observed",
            },
        ]
    ).to_csv(path, index=False)

    result = load_mspd_cohorts(path)

    assert len(result) == 1
    assert result.loc[0, "quarter"] == "2024Q1"
    assert result.loc[0, "cohort_id"] == "cohort-1"
    assert result.loc[0, "outstanding"] == 220.0


def test_enrich_mspd_cohorts_with_auction_frn_and_tips_sources(tmp_path):
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2024Q3",
                "cohort_id": "FRN1",
                "cusip": "FRN1",
                "security_type": "frn",
                "issue_date": "2024-07-31",
                "maturity_date": "2026-07-31",
                "outstanding": 100.0,
            },
            {
                "quarter": "2024Q3",
                "cohort_id": "TIPS1",
                "cusip": "TIPS1",
                "security_type": "tips",
                "issue_date": "2020-07-15",
                "maturity_date": "2030-07-15",
                "outstanding": 100.0,
            },
        ]
    )
    auction_path = tmp_path / "auctions.csv"
    pd.DataFrame(
        [
            {
                "cusip": "FRN1",
                "issue_date": "2024-07-31",
                "maturity_date": "2026-07-31",
                "security_term": "2-Year",
                "price_per100": 99.75,
                "avg_med_yield": 5.0,
                "int_rate": 0.0,
                "spread": 0.182,
                "frn_index_determination_rate": 5.25,
            },
            {
                "cusip": "TIPS1",
                "issue_date": "2020-07-15",
                "maturity_date": "2030-07-15",
                "security_term": "10-Year",
                "price_per100": 101.5,
                "avg_med_yield": 1.2,
                "int_rate": 0.125,
                "index_ratio_on_issue_date": 1.01,
                "ref_cpi_on_issue_date": 250.0,
            },
        ]
    ).to_csv(auction_path, index=False)
    frn_path = tmp_path / "frn.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-09-29",
                "start_of_accrual_period": "2024-09-29",
                "cusip": "FRN1",
                "spread": 0.182,
                "daily_index": 5.2,
                "daily_accrued_int_per100": 0.10,
                "accr_int_per100_pmt_period": 0.40,
            },
            {
                "record_date": "2024-09-30",
                "start_of_accrual_period": "2024-09-30",
                "cusip": "FRN1",
                "spread": 0.182,
                "daily_index": 5.3,
                "daily_accrued_int_per100": 0.20,
                "accr_int_per100_pmt_period": 0.45,
            },
        ]
    ).to_csv(frn_path, index=False)
    tips_path = tmp_path / "tips.csv"
    pd.DataFrame(
        [
            {
                "cusip": "TIPS1",
                "index_date": "2024-09-30",
                "ref_cpi": 310.0,
                "index_ratio": 1.24,
            }
        ]
    ).to_csv(tips_path, index=False)

    enriched = enrich_mspd_cohorts_with_security_sources(
        cohorts,
        auction_path=auction_path,
        frn_daily_indexes_path=frn_path,
        tips_cpi_path=tips_path,
        start_quarter="2024Q3",
        end_quarter="2024Q3",
    )

    frn = enriched.set_index("cohort_id").loc["FRN1"]
    assert frn["issue_price_ratio"] == pytest.approx(0.9975)
    assert frn["issue_yield_decimal"] == pytest.approx(0.05)
    assert frn["FixedSpread"] == pytest.approx(0.00182)
    assert frn["BenchmarkRate_FRN"] == pytest.approx(0.053)
    assert frn["AccruedInterest_FRN_Per100"] == pytest.approx(0.30)

    tips = enriched.set_index("cohort_id").loc["TIPS1"]
    assert tips["coupon_rate_decimal"] == pytest.approx(0.00125)
    assert tips["IndexRatio"] == pytest.approx(1.24)
    assert tips["ReferenceCPI_Issue"] == pytest.approx(250.0)
    assert tips["ReferenceCPI_End"] == pytest.approx(310.0)
    assert "tips_cpi_detail_joined" in tips["security_enrichment_status"]


def test_load_tips_inflation_adjustment_stocks_filters_and_normalizes(tmp_path):
    path = tmp_path / "mspd.csv"
    pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "security_class1_desc": "Inflation-Protected Securities",
                "security_class2_desc": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt": 1000.0,
                "inflation_adj_amt": 120.0,
                "redeemed_amt": 0.0,
                "outstanding_amt": 1120.0,
            },
            {
                "record_date": "2025-01-31",
                "security_class1_desc": "Inflation-Protected Securities",
                "security_class2_desc": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt": 1000.0,
                "inflation_adj_amt": 125.0,
                "redeemed_amt": 0.0,
                "outstanding_amt": 1125.0,
            },
            {
                "record_date": "2025-01-31",
                "security_class1_desc": "Bills",
                "security_class2_desc": "912796AA1",
                "issue_date": "2024-10-31",
                "maturity_date": "2025-01-31",
                "issued_amt": 500.0,
                "inflation_adj_amt": 0.0,
                "redeemed_amt": 500.0,
                "outstanding_amt": 0.0,
            },
            {
                "record_date": "2025-01-31",
                "security_class1_desc": "Total Marketable",
                "security_class2_desc": "Total",
                "issue_date": "2024-10-31",
                "maturity_date": "2025-01-31",
                "issued_amt": 1500.0,
                "inflation_adj_amt": 125.0,
                "redeemed_amt": 500.0,
                "outstanding_amt": 1125.0,
            },
        ]
    ).to_csv(path, index=False)

    result = load_tips_inflation_adjustment_stocks(path, start_quarter="2025Q1", end_quarter="2025Q1")

    assert result["record_date"].tolist() == ["2024-12-31", "2025-01-31"]
    assert result["quarter"].tolist() == ["2024Q4", "2025Q1"]
    assert result["cusip"].tolist() == ["912810AA1", "912810AA1"]
    assert result["inflation_adjustment_mil"].tolist() == pytest.approx([120.0, 125.0])
    assert set(result["source_status"]) == {"mspd_monthly_tips_inflation_adjustment_row"}


def test_load_tips_auction_events_and_reference_path(tmp_path):
    auctions = tmp_path / "auctions.csv"
    pd.DataFrame(
        [
            {
                "cusip": "912810AA1",
                "security_type": "TIPS",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "maturity_date": "2035-01-15",
                "total_accepted": 1_000_000_000.0,
                "int_rate": 1.25,
                "index_ratio_on_issue_date": 1.05,
                "ref_cpi_on_dated_date": 300.0,
                "ref_cpi_on_issue_date": 315.0,
                "adj_accrued_int_per1000": 1.2,
            },
            {
                "cusip": "912810BB2",
                "security_type": "Note",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "maturity_date": "2035-01-15",
                "total_accepted": 2_000_000_000.0,
            },
        ]
    ).to_csv(auctions, index=False)
    cpi = tmp_path / "tips_cpi.csv"
    pd.DataFrame(
        [
            {"cusip": "912810AA1", "index_date": "2025-01-15", "ref_cpi": 315.0, "index_ratio": 1.05},
            {"cusip": "912810CC3", "index_date": "2025-01-15", "ref_cpi": 315.0, "index_ratio": 1.04},
        ]
    ).to_csv(cpi, index=False)

    events = load_tips_auction_events(auctions)
    ref = load_tips_cpi_reference_path(cpi)

    assert len(events) == 1
    assert events.loc[0, "total_accepted_mil"] == pytest.approx(1000.0)
    assert events.loc[0, "coupon_rate_decimal"] == pytest.approx(0.0125)
    assert events.loc[0, "ref_cpi_on_dated_date"] == pytest.approx(300.0)
    assert ref["index_date"].tolist() == ["2025-01-15"]
    assert ref["ref_cpi"].tolist() == pytest.approx([315.0])
