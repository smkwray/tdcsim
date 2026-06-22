from __future__ import annotations

import pandas as pd

from historical_replay_auction import (
    AUCTION_ALLOTMENT_PROXY_COLUMNS,
    build_auction_absorption_reconciliation,
    build_auction_holder_prior_panel,
    load_auction_allotment_proxy,
    write_auction_allotment_proxy,
)


def _write_sources(
    tmp_path,
    *,
    allotment_rows: list[dict[str, object]],
    auction_rows: list[dict[str, object]] | None = None,
) -> tuple:
    allotment_path = tmp_path / "auction_allotment_panel_base_slim.csv"
    pd.DataFrame(allotment_rows).to_csv(allotment_path, index=False)
    auction_path = None
    if auction_rows is not None:
        auction_path = tmp_path / "auctions_query.csv"
        pd.DataFrame(auction_rows).to_csv(auction_path, index=False)
    return allotment_path, auction_path


def test_load_auction_allotment_proxy_normalizes_issue_quarter_and_filters_range(tmp_path):
    allotment_path, auction_path = _write_sources(
        tmp_path,
        allotment_rows=[
            {
                "cusip": "912345AA1",
                "auction_date": "2024-03-29",
                "issue_date": "2024-04-01",
                "maturity_date": "2024-07-01",
                "security_type": "Bill",
                "security_term": "13-Week",
                "raw_investor_class": "dealers and brokers",
                "narrow_investor_class": "dealers_brokers",
                "broad_investor_class": "dealers",
                "is_bridge_class": "True",
                "allotment_amount": 60.0,
                "allotment_total_clean": 100.0,
                "allotment_share_clean": 0.60,
                "accepted_amount": 101.0,
                "offering_amount": 95.0,
            },
            {
                "cusip": "912345BB2",
                "auction_date": "2024-01-10",
                "issue_date": "2024-01-15",
                "maturity_date": "2025-01-15",
                "security_type": "Note",
                "security_term": "1-Year",
                "raw_investor_class": "investment funds",
                "narrow_investor_class": "investment_funds",
                "broad_investor_class": "investment_funds",
                "is_bridge_class": "False",
                "allotment_amount": 25.0,
                "allotment_total_clean": 80.0,
                "allotment_share_clean": 0.3125,
                "accepted_amount": 80.0,
                "offering_amount": 75.0,
            },
        ],
    )

    result = load_auction_allotment_proxy(
        allotment_panel_path=allotment_path,
        auction_terms_path=None,
        start_quarter="2024Q2",
    )

    assert result["quarter"].tolist() == ["2024Q2"]
    assert result.loc[0, "auction_date"] == "2024-03-29"
    assert result.loc[0, "issue_date"] == "2024-04-01"
    assert result.loc[0, "reconciliation_gap"] == 1.0
    assert bool(result.loc[0, "is_bridge_class"]) is True
    assert result.loc[0, "source_status"] == "buycurve_allotment_panel_observed"
    assert result.loc[0, "evidence_label"] == "observed"


def test_load_auction_allotment_proxy_uses_fiscaldata_fallback_for_missing_terms(tmp_path):
    allotment_path, auction_path = _write_sources(
        tmp_path,
        allotment_rows=[
            {
                "cusip": "912345CC3",
                "auction_date": "2024-06-26",
                "issue_date": "2024-07-01",
                "maturity_date": "2024-09-30",
                "security_type": "",
                "security_term": "",
                "raw_investor_class": "foreign and international",
                "narrow_investor_class": "foreign_international",
                "broad_investor_class": "foreign_international",
                "is_bridge_class": "False",
                "allotment_amount": 20.0,
                "allotment_total_clean": 50.0,
                "allotment_share_clean": 0.40,
                "accepted_amount": "",
                "offering_amount": "",
            }
        ],
        auction_rows=[
            {
                "cusip": "912345CC3",
                "auction_date": "2024-06-26",
                "issue_date": "2024-07-01",
                "maturity_date": "2024-09-30",
                "security_type": "Bill",
                "security_term": "13-Week",
                "total_accepted": 52.0,
                "offering_amt": 50.0,
            }
        ],
    )

    result = load_auction_allotment_proxy(
        allotment_panel_path=allotment_path,
        auction_terms_path=auction_path,
    )

    assert result.loc[0, "quarter"] == "2024Q3"
    assert result.loc[0, "security_type"] == "Bill"
    assert result.loc[0, "security_term"] == "13-Week"
    assert result.loc[0, "accepted_amount"] == 52.0
    assert result.loc[0, "offering_amount"] == 50.0
    assert result.loc[0, "reconciliation_gap"] == 2.0
    assert result.loc[0, "source_status"] == "buycurve_allotment_panel_with_fiscaldata_fallback"


def test_write_auction_allotment_proxy_emits_expected_columns(tmp_path):
    allotment_path, auction_path = _write_sources(
        tmp_path,
        allotment_rows=[
            {
                "cusip": "912345DD4",
                "auction_date": "2024-09-25",
                "issue_date": "2024-09-30",
                "maturity_date": "2026-09-30",
                "security_type": "Note",
                "security_term": "2-Year",
                "raw_investor_class": "individuals",
                "narrow_investor_class": "individuals",
                "broad_investor_class": "individuals",
                "is_bridge_class": "False",
                "allotment_amount": 30.0,
                "allotment_total_clean": 90.0,
                "allotment_share_clean": 1 / 3,
                "accepted_amount": 91.0,
                "offering_amount": 88.0,
            }
        ],
    )
    output_path = tmp_path / "auction_allotment_proxy.csv"

    written = write_auction_allotment_proxy(
        output_path=output_path,
        allotment_panel_path=allotment_path,
        auction_terms_path=None,
    )
    reloaded = pd.read_csv(output_path)

    assert written.columns.tolist() == list(AUCTION_ALLOTMENT_PROXY_COLUMNS)
    assert reloaded.columns.tolist() == list(AUCTION_ALLOTMENT_PROXY_COLUMNS)
    assert reloaded.loc[0, "reconciliation_gap"] == 1.0


def test_auction_absorption_reconciliation_dedupes_auction_totals_and_signs_private():
    proxy = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cusip": "912345AA1",
                "auction_date": "2025-01-28",
                "issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "security_type": "Note",
                "security_term": "2-Year",
                "raw_investor_class": "dealers",
                "narrow_investor_class": "dealers_brokers",
                "broad_investor_class": "dealers",
                "is_bridge_class": True,
                "allotment_amount": 60_000_000.0,
                "allotment_total_clean": 100_000_000.0,
                "allotment_share_clean": 0.60,
                "accepted_amount": 101_000_000.0,
                "offering_amount": 100_000_000.0,
                "reconciliation_gap": 1_000_000.0,
                "source_status": "observed",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cusip": "912345AA1",
                "auction_date": "2025-01-28",
                "issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "security_type": "Note",
                "security_term": "2-Year",
                "raw_investor_class": "investment funds",
                "narrow_investor_class": "investment_funds",
                "broad_investor_class": "investment_funds",
                "is_bridge_class": False,
                "allotment_amount": 30_000_000.0,
                "allotment_total_clean": 100_000_000.0,
                "allotment_share_clean": 0.30,
                "accepted_amount": 101_000_000.0,
                "offering_amount": 100_000_000.0,
                "reconciliation_gap": 1_000_000.0,
                "source_status": "observed",
                "evidence_label": "observed",
            },
            {
                "quarter": "2025Q1",
                "cusip": "912345AA1",
                "auction_date": "2025-01-28",
                "issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "security_type": "Note",
                "security_term": "2-Year",
                "raw_investor_class": "foreign",
                "narrow_investor_class": "foreign_international",
                "broad_investor_class": "foreign_international",
                "is_bridge_class": False,
                "allotment_amount": 10_000_000.0,
                "allotment_total_clean": 100_000_000.0,
                "allotment_share_clean": 0.10,
                "accepted_amount": 101_000_000.0,
                "offering_amount": 100_000_000.0,
                "reconciliation_gap": 1_000_000.0,
                "source_status": "observed",
                "evidence_label": "observed",
            },
        ]
    )

    result = build_auction_absorption_reconciliation(proxy)

    by_class = result.set_index("broad_investor_class")
    assert by_class.loc["investment_funds", "tdcsim_holder"] == "Private"
    assert bool(by_class.loc["investment_funds", "included_in_tdc_auction_absorption"]) is True
    assert by_class.loc["investment_funds", "signed_tdc_auction_absorption_mil"] == -30.0
    assert bool(by_class.loc["foreign_international", "included_in_identified_primary_allotment"]) is True
    assert bool(by_class.loc["foreign_international", "included_in_tdc_auction_absorption"]) is False
    assert by_class.loc["dealers", "tdc_absorption_role"] == "dealer_bridge_not_final_holder"
    assert bool(by_class.loc["dealers", "included_in_identified_primary_allotment"]) is False
    assert result["unique_auction_accepted_amount_mil"].nunique() == 1
    assert result["unique_auction_accepted_amount_mil"].iloc[0] == 101.0
    assert result["quarter_allotment_reconciliation_gap_mil"].iloc[0] == 1.0


def test_auction_holder_prior_panel_excludes_bridge_classes():
    proxy = pd.DataFrame(
        [
            {
                "cusip": "912345AA1",
                "issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "broad_investor_class": "dealers",
                "is_bridge_class": True,
                "allotment_amount": 60_000_000.0,
            },
            {
                "cusip": "912345AA1",
                "issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "broad_investor_class": "banks",
                "is_bridge_class": False,
                "allotment_amount": 10_000_000.0,
            },
            {
                "cusip": "912345AA1",
                "issue_date": "2025-01-31",
                "maturity_date": "2027-01-31",
                "broad_investor_class": "investment_funds",
                "is_bridge_class": False,
                "allotment_amount": 30_000_000.0,
            },
        ]
    )

    result = build_auction_holder_prior_panel(proxy)

    assert len(result.index) == 1
    assert result.loc[0, "prior_holder_Banks"] == 10_000_000.0
    assert result.loc[0, "prior_holder_Private"] == 30_000_000.0
    assert result.loc[0, "prior_holder_CB"] == 0.0
    assert result.loc[0, "prior_holder_Foreign"] == 0.0
    assert result.loc[0, "auction_prior_total_amount"] == 40_000_000.0
    assert result.loc[0, "auction_prior_holder_count"] == 2
