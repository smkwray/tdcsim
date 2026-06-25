from __future__ import annotations

import pandas as pd
import pytest

from historical_replay_materializer import materialize_portfolio
from tdc_shared import (
    BOND_PORTFOLIO_COLS,
    PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
    PRIVATE_SUBBUCKET_MMF,
)


def test_materializer_maps_holder_routes_and_reaggregates_allocations():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "bill_a",
                "outstanding": 60.0,
                "SecurityType": "Fixed",
                "IssueDate": pd.Timestamp("2024-12-15"),
                "MaturityDate": pd.Timestamp("2025-06-15"),
                "OriginalMaturityYears": 0.5,
                "CouponRate": 0.0,
                "MaturityCategory": "bills",
                "IssuePriceRatio": 0.99,
                "IssueProceeds": 59.4,
                "IssueYieldAtIssue": 0.04,
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "tips_b",
                "outstanding": 40.0,
                "SecurityType": "TIPS",
                "IssueDate": pd.Timestamp("2023-01-15"),
                "MaturityDate": pd.Timestamp("2028-01-15"),
                "OriginalMaturityYears": 5.0,
                "CouponRate": 0.01,
                "MaturityCategory": "tips",
                "OriginalPrincipal": 35.0,
                "AdjustedPrincipal": 40.0,
                "ReferenceCPI_Issue": 280.0,
                "IndexRatio": 1.142857142857,
                "IssuePriceRatio": 1.01,
                "IssueProceeds": 40.4,
                "IssueYieldAtIssue": 0.012,
            },
        ]
    )
    allocations = pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "cohort_id": "bill_a", "allocated_outstanding": 15.0},
            {"quarter": "2025Q1", "sector": "MMF", "cohort_id": "bill_a", "allocated_outstanding": 10.0},
            {"quarter": "2025Q1", "sector": "DomesticNonbankExMMF", "cohort_id": "bill_a", "allocated_outstanding": 20.0},
            {"quarter": "2025Q1", "sector": "Other", "cohort_id": "bill_a", "allocated_outstanding": 15.0},
            {"quarter": "2025Q1", "sector": "Foreign", "cohort_id": "tips_b", "allocated_outstanding": 25.0},
            {"quarter": "2025Q1", "sector": "MMF", "cohort_id": "tips_b", "allocated_outstanding": 5.0},
            {"quarter": "2025Q1", "sector": "TrustFunds", "cohort_id": "tips_b", "allocated_outstanding": 10.0},
            {"quarter": "2025Q1", "sector": "foreign_international", "cohort_id": "tips_b", "allocated_outstanding": 2.0},
            {"quarter": "2025Q1", "sector": "money_market_cash", "cohort_id": "tips_b", "allocated_outstanding": 3.0},
        ]
    )

    portfolio = materialize_portfolio(allocations, cohorts, start_bond_id=10)

    assert all(col in portfolio.columns for col in BOND_PORTFOLIO_COLS)
    assert portfolio["BondID"].tolist() == [10, 11, 12, 13, 14, 15, 16, 17, 18]

    mmf_rows = portfolio[portfolio["source_sector"].isin(["MMF", "money_market_cash"])]
    assert set(mmf_rows["HolderType"].astype(str)) == {"Private"}
    assert set(mmf_rows["HolderSubBucket"].astype(str)) == {PRIVATE_SUBBUCKET_MMF}

    foreign_rows = portfolio[portfolio["source_sector"].isin(["Foreign", "foreign_international"])]
    assert set(foreign_rows["HolderType"].astype(str)) == {"Foreign"}

    dnx_rows = portfolio[portfolio["source_sector"].isin(["DomesticNonbankExMMF", "Other"])]
    assert set(dnx_rows["HolderType"].astype(str)) == {"Private"}
    assert set(dnx_rows["HolderSubBucket"].astype(str)) == {PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}

    by_source_sector = portfolio.groupby("source_sector")["FaceValue"].sum().sort_index()
    expected_by_source_sector = allocations.groupby("sector")["allocated_outstanding"].sum().sort_index()
    pd.testing.assert_series_equal(
        by_source_sector,
        expected_by_source_sector,
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )

    by_cohort = portfolio.groupby("cohort_id")["FaceValue"].sum().sort_index()
    expected_by_cohort = allocations.groupby("cohort_id")["allocated_outstanding"].sum().sort_index()
    pd.testing.assert_series_equal(
        by_cohort,
        expected_by_cohort,
        atol=1e-9,
        rtol=0.0,
        check_names=False,
    )

    tips_rows = portfolio[portfolio["SecurityType"] == "TIPS"].sort_values("BondID")
    assert not tips_rows.empty
    assert abs(tips_rows["AdjustedPrincipal"].sum() - 45.0) <= 1e-9
    assert set(portfolio["quarter"].astype(str)) == {"2025Q1"}
    identity_gap = tips_rows["AdjustedPrincipal"] - tips_rows["OriginalPrincipal"] * tips_rows["IndexRatio"]
    assert identity_gap.abs().max() <= 1e-9


def test_materializer_normalizes_raw_security_units_and_preserves_sector_labels():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "912345AA1",
                "cusip": "912345AA1",
                "outstanding": 100.0,
                "security_type": "note",
                "issue_date": "2024-01-15",
                "maturity_date": "2029-01-15",
                "original_maturity_years": 5.0,
                "interest_rate_pct": 4.5,
                "yield_pct": 4.25,
                "price_per100": 99.5,
            }
        ]
    )
    allocations = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "sector": "household_sector",
                "native_sector": "household_sector",
                "broad_holder_class": "individuals",
                "tdcsim_holder": "Private",
                "tdcsim_holder_subbucket": PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
                "cohort_id": "912345AA1",
                "allocated_outstanding": 40.0,
            }
        ]
    )

    portfolio = materialize_portfolio(allocations, cohorts)

    row = portfolio.iloc[0]
    assert row["SecurityType"] == "Fixed"
    assert row["CouponRate"] == pytest.approx(0.045)
    assert row["IssueYieldAtIssue"] == pytest.approx(0.0425)
    assert row["IssuePriceRatio"] == pytest.approx(0.995)
    assert row["IssueProceeds"] == pytest.approx(39.8)
    assert row["native_sector"] == "household_sector"
    assert row["broad_holder_class"] == "individuals"
    assert row["tdcsim_holder"] == "Private"


def test_materializer_preserves_source_basis_residual_holder():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "residual_cohort",
                "cusip": "912345AA1",
                "outstanding": 100.0,
                "security_type": "note",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "original_maturity_years": 10.0,
            }
        ]
    )
    allocations = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "sector": "MSPD_Z1_SourceBasisResidual",
                "native_sector": "MSPD_Z1_SourceBasisResidual",
                "broad_holder_class": "source_basis_residual",
                "tdcsim_holder": "source_basis_residual",
                "tdcsim_holder_subbucket": "",
                "cohort_id": "residual_cohort",
                "allocated_outstanding": 25.0,
            }
        ]
    )

    portfolio = materialize_portfolio(allocations, cohorts)

    assert portfolio.loc[0, "HolderType"] == "SourceBasisResidual"
    assert portfolio.loc[0, "tdcsim_holder"] == "SourceBasisResidual"
    assert portfolio.loc[0, "HolderSubBucket"] == ""
    assert portfolio.loc[0, "source_sector"] == "MSPD_Z1_SourceBasisResidual"


def test_materializer_does_not_use_tips_inflation_adjustment_as_principal():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "tips_c",
                "cusip": "912345TI1",
                "outstanding": 100.0,
                "security_type": "tips",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "original_maturity_years": 10.0,
                "inflation_adj_amt": 12.0,
                "IndexRatio": 1.2,
            },
            {
                "quarter": "2025Q1",
                "cohort_id": "note_c",
                "cusip": "912345NO1",
                "outstanding": 80.0,
                "security_type": "note",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "original_maturity_years": 10.0,
                "inflation_adj_amt": 99.0,
            },
        ]
    )
    allocations = pd.DataFrame(
        [
            {"quarter": "2025Q1", "sector": "Banks", "cohort_id": "tips_c", "allocated_outstanding": 30.0},
            {"quarter": "2025Q1", "sector": "Foreign", "cohort_id": "tips_c", "allocated_outstanding": 70.0},
            {"quarter": "2025Q1", "sector": "Banks", "cohort_id": "note_c", "allocated_outstanding": 20.0},
        ]
    )

    portfolio = materialize_portfolio(allocations, cohorts)

    tips_rows = portfolio[portfolio["cohort_id"] == "tips_c"]
    note_row = portfolio[portfolio["cohort_id"] == "note_c"].iloc[0]
    assert tips_rows["AdjustedPrincipal"].sum() == pytest.approx(100.0)
    assert tips_rows["FaceValue"].sum() == pytest.approx(100.0)
    assert tips_rows["OriginalPrincipal"].sum() == pytest.approx(100.0 / 1.2)
    assert note_row["AdjustedPrincipal"] == pytest.approx(20.0)
    assert note_row["OriginalPrincipal"] == pytest.approx(0.0)


def test_materializer_first_interest_schedule_handles_missing_maturity():
    cohorts = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cohort_id": "console_like",
                "cusip": "912345AA1",
                "outstanding": 100.0,
                "security_type": "note",
                "issue_date": "2025-01-15",
                "maturity_date": pd.NaT,
                "original_maturity_years": 10.0,
                "coupon_rate": 0.04,
                "interest_pay_date_1": "2025-07-15",
                "interest_pay_date_2": "2026-01-15",
            }
        ]
    )
    allocations = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "sector": "Banks",
                "cohort_id": "console_like",
                "allocated_outstanding": 25.0,
            }
        ]
    )

    portfolio = materialize_portfolio(allocations, cohorts)

    assert portfolio.loc[0, "FirstInterestPaymentDate"] == pd.Timestamp("2025-07-15")
    assert portfolio.loc[0, "InterestPaymentFrequency"] == pytest.approx(2.0)
