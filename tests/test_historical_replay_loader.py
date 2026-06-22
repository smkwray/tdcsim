"""Tests for historical replay CSV loaders."""

from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_loader import (
    enrich_mspd_cohorts_with_security_sources,
    load_auction_flows,
    load_mspd_cohorts,
    load_quarterly_cash,
    load_sector_positions,
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
                "cusip": "FRN1",
                "spread": 0.182,
                "daily_index": 5.2,
                "accr_int_per100_pmt_period": 0.40,
            },
            {
                "record_date": "2024-09-30",
                "cusip": "FRN1",
                "spread": 0.182,
                "daily_index": 5.3,
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
    assert frn["AccruedInterest_FRN_Per100"] == pytest.approx(0.45)

    tips = enriched.set_index("cohort_id").loc["TIPS1"]
    assert tips["coupon_rate_decimal"] == pytest.approx(0.00125)
    assert tips["IndexRatio"] == pytest.approx(1.24)
    assert tips["ReferenceCPI_Issue"] == pytest.approx(250.0)
    assert tips["ReferenceCPI_End"] == pytest.approx(310.0)
    assert "tips_cpi_detail_joined" in tips["security_enrichment_status"]
