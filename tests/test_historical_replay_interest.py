from __future__ import annotations

import pandas as pd
import pytest

from historical_replay import _build_interest_proxy_alignment
from historical_replay_interest import (
    DETAIL_COLUMNS,
    aggregate_stock_only_interest,
    align_stock_only_interest_proxy,
    build_quarterly_interest_detail,
    build_treasury_interest_expense_diagnostic,
)
from tdc_shared import PRIVATE_SUBBUCKET_DOMESTIC_NONBANK, PRIVATE_SUBBUCKET_MMF


@pytest.fixture
def quarterly_portfolio_snapshots() -> dict[str, pd.DataFrame]:
    return {
        "2025Q1": pd.DataFrame(
            [
                {
                    "quarter": "2025Q1",
                    "cohort_id": "NOTE1",
                    "source_sector": "banks",
                    "HolderType": "Banks",
                    "SecurityType": "Fixed",
                    "MaturityCategory": "notes",
                    "IssueDate": pd.Timestamp("2024-01-15"),
                    "MaturityDate": pd.Timestamp("2027-01-15"),
                    "OriginalMaturityYears": 3.0,
                    "FaceValue": 100.0,
                    "CouponRate": 0.04,
                    "IssuePriceRatio": 1.0,
                    "IssueProceeds": 100.0,
                    "IssueYieldAtIssue": 0.04,
                },
                {
                    "quarter": "2025Q1",
                    "cohort_id": "BILL1",
                    "source_sector": "MMF",
                    "HolderType": "Private",
                    "HolderSubBucket": PRIVATE_SUBBUCKET_MMF,
                    "SecurityType": "Fixed",
                    "MaturityCategory": "bills",
                    "IssueDate": pd.Timestamp("2025-02-01"),
                    "MaturityDate": pd.Timestamp("2025-05-01"),
                    "OriginalMaturityYears": 0.25,
                    "FaceValue": 100.0,
                    "CouponRate": 0.0,
                    "IssuePriceRatio": 0.98,
                    "IssueProceeds": 98.0,
                    "IssueYieldAtIssue": 0.08,
                },
                {
                    "quarter": "2025Q1",
                    "cohort_id": "FRN1",
                    "source_sector": "foreign_international",
                    "HolderType": "Foreign",
                    "SecurityType": "FRN",
                    "IssueDate": pd.Timestamp("2024-12-15"),
                    "MaturityDate": pd.Timestamp("2026-12-15"),
                    "FaceValue": 80.0,
                    "FixedSpread": 0.0025,
                    "AccruedInterest_FRN": 1.25,
                },
                {
                    "quarter": "2025Q1",
                    "cohort_id": "TIPS1",
                    "source_sector": "DomesticNonbankExMMF",
                    "HolderType": "Private",
                    "HolderSubBucket": PRIVATE_SUBBUCKET_DOMESTIC_NONBANK,
                    "SecurityType": "TIPS",
                    "IssueDate": pd.Timestamp("2023-01-15"),
                    "MaturityDate": pd.Timestamp("2028-01-15"),
                    "FaceValue": 100.0,
                    "CouponRate": 0.01,
                    "OriginalPrincipal": 100.0,
                    "AdjustedPrincipal": 110.0,
                    "ReferenceCPI_Issue": 250.0,
                    "IndexRatio": 1.10,
                    "IssuePriceRatio": 1.01,
                    "IssueProceeds": 101.0,
                    "IssueYieldAtIssue": 0.012,
                },
            ]
        )
    }


@pytest.fixture
def interest_detail(quarterly_portfolio_snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return build_quarterly_interest_detail(quarterly_portfolio_snapshots)


def test_build_quarterly_interest_detail_emits_expected_components_and_math(
    interest_detail: pd.DataFrame,
):
    assert interest_detail.columns.tolist() == DETAIL_COLUMNS
    assert set(interest_detail["component"]) == {
        "bill_discount_amortization",
        "fixed_coupon_accrual",
        "frn_accrued_interest_pass_through",
        "tips_coupon_accrual",
    }

    fixed_row = interest_detail.loc[interest_detail["component"] == "fixed_coupon_accrual"].iloc[0]
    assert fixed_row["broad_holder_class"] == "Banks"
    assert fixed_row["tdcsim_holder"] == "Banks"
    assert fixed_row["days_held"] == 90
    assert fixed_row["face_begin"] == pytest.approx(100.0)
    assert fixed_row["face_end"] == pytest.approx(100.0)
    assert fixed_row["interest_amount"] == pytest.approx(100.0 * 0.04 * 90.0 / 365.25, rel=1e-9)

    bill_row = interest_detail.loc[interest_detail["component"] == "bill_discount_amortization"].iloc[0]
    assert bill_row["tdcsim_holder"] == f"Private:{PRIVATE_SUBBUCKET_MMF}"
    assert bill_row["days_held"] == 59
    assert bill_row["exposure_fraction"] == pytest.approx(59.0 / 90.0, rel=1e-9)
    assert bill_row["face_begin"] == pytest.approx(0.0)
    assert bill_row["face_end"] == pytest.approx(100.0)
    assert bill_row["interest_amount"] == pytest.approx(2.0 * 59.0 / 90.0, rel=1e-9)

    frn_row = interest_detail.loc[
        interest_detail["component"] == "frn_accrued_interest_pass_through"
    ].iloc[0]
    assert frn_row["broad_holder_class"] == "Foreign"
    assert frn_row["fixed_spread"] == pytest.approx(0.0025)
    assert frn_row["interest_amount"] == pytest.approx(1.25)

    tips_row = interest_detail.loc[interest_detail["component"] == "tips_coupon_accrual"].iloc[0]
    assert tips_row["tdcsim_holder"] == f"Private:{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}"
    assert tips_row["exposure_base"] == pytest.approx(110.0)
    assert tips_row["index_ratio_start"] == pytest.approx(1.10)
    assert tips_row["index_ratio_end"] == pytest.approx(1.10)
    assert tips_row["ref_cpi_start"] == pytest.approx(275.0)
    assert tips_row["ref_cpi_end"] == pytest.approx(275.0)
    assert tips_row["interest_amount"] == pytest.approx(110.0 * 0.01 * 90.0 / 365.25, rel=1e-9)


def test_alignment_helper_aggregates_proxy_by_holder_and_compares_to_reference(
    interest_detail: pd.DataFrame,
):
    tdcsim_holder_proxy = aggregate_stock_only_interest(interest_detail)
    assert set(tdcsim_holder_proxy["tdcsim_holder"]) == {
        "Banks",
        "Foreign",
        f"Private:{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}",
        f"Private:{PRIVATE_SUBBUCKET_MMF}",
    }

    reference = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "bank_interest_point": 100.0 * 0.04 * 90.0 / 365.25,
                "foreign_interest_point": 1.00,
                "private_interest_point": (2.0 * 59.0 / 90.0) + (110.0 * 0.01 * 90.0 / 365.25),
            }
        ]
    )

    aligned = align_stock_only_interest_proxy(
        interest_detail,
        reference=reference,
        holder_column="broad_holder_class",
        reference_mapping={
            "Banks": "bank_interest_point",
            "Foreign": "foreign_interest_point",
            "Private": "private_interest_point",
        },
    )

    by_holder = aligned.set_index("broad_holder_class")
    assert by_holder.loc["Banks", "stock_only_interest_proxy"] == pytest.approx(
        100.0 * 0.04 * 90.0 / 365.25,
        rel=1e-9,
    )
    assert by_holder.loc["Private", "stock_only_interest_proxy"] == pytest.approx(
        (2.0 * 59.0 / 90.0) + (110.0 * 0.01 * 90.0 / 365.25),
        rel=1e-9,
    )
    assert by_holder.loc["Foreign", "difference_vs_reference"] == pytest.approx(0.25, rel=1e-9)


def test_interest_alignment_reports_stock_and_projection_tolerance_separately(
    interest_detail: pd.DataFrame,
):
    tdc_panel = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "banks_plus_credit_union_tier2_candidate_interest_mil": 1.2,
                "banks_plus_credit_union_tier2_candidate_interest_low_mil": 1.2,
                "banks_plus_credit_union_tier2_candidate_interest_high_mil": 1.2,
            }
        ]
    )

    aligned = _build_interest_proxy_alignment(interest_detail, tdc_panel)
    bank = aligned.loc[aligned["native_sector"] == "Banks"].iloc[0]

    assert bank["stock_only_proxy"] != pytest.approx(1.2)
    assert bank["interest_constrained_proxy"] == pytest.approx(1.2)
    assert bool(bank["stock_only_within_tolerance"]) is False
    assert bool(bank["constrained_within_tolerance"]) is True
    assert bool(bank["within_tolerance"]) is True


def test_source_basis_residual_interest_rows_are_excluded_from_default_proxy():
    detail = build_quarterly_interest_detail(
        {
            "2025Q1": pd.DataFrame(
                [
                    {
                        "quarter": "2025Q1",
                        "cohort_id": "NOTE1",
                        "source_sector": "banks",
                        "HolderType": "Banks",
                        "tdcsim_holder": "Banks",
                        "SecurityType": "Fixed",
                        "MaturityCategory": "notes",
                        "IssueDate": pd.Timestamp("2024-01-15"),
                        "MaturityDate": pd.Timestamp("2027-01-15"),
                        "FaceValue": 100.0,
                        "CouponRate": 0.04,
                        "IssuePriceRatio": 1.0,
                        "IssueProceeds": 100.0,
                        "IssueYieldAtIssue": 0.04,
                    },
                    {
                        "quarter": "2025Q1",
                        "cohort_id": "NOTE1",
                        "source_sector": "MSPD_Z1_SourceBasisResidual",
                        "HolderType": "SourceBasisResidual",
                        "tdcsim_holder": "SourceBasisResidual",
                        "SecurityType": "Fixed",
                        "MaturityCategory": "notes",
                        "IssueDate": pd.Timestamp("2024-01-15"),
                        "MaturityDate": pd.Timestamp("2027-01-15"),
                        "FaceValue": 20.0,
                        "CouponRate": 0.04,
                        "IssuePriceRatio": 1.0,
                        "IssueProceeds": 20.0,
                        "IssueYieldAtIssue": 0.04,
                    },
                ]
            )
        }
    )

    residual = detail[detail["tdcsim_holder"] == "SourceBasisResidual"]
    assert not residual.empty
    assert residual["excluded_from_default_canonical"].all()

    proxy = aggregate_stock_only_interest(detail)
    assert set(proxy["tdcsim_holder"]) == {"Banks"}
    proxy_with_excluded = aggregate_stock_only_interest(detail, include_excluded=True)
    assert set(proxy_with_excluded["tdcsim_holder"]) == {"Banks", "SourceBasisResidual"}


def test_treasury_interest_expense_diagnostic_maps_official_pools(interest_detail: pd.DataFrame):
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED DISCOUNT",
                "expense_type_desc": "Treasury Bills",
                "month_expense_amt": 2_000_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Treasury Notes",
                "month_expense_amt": 3_000_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Treasury Floating Rate Notes (FRN)",
                "month_expense_amt": 4_000_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Inflation Protected Securities (TIPS)",
                "month_expense_amt": 5_000_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Int. Expense Inflation Compensation (TIPS)",
                "month_expense_amt": 6_000_000.0,
            },
        ]
    )
    candidate = pd.DataFrame(
        [
            {
                "date": "2025-03-31",
                "component_family": "bill_discount",
                "component_anchored_interest_mil": 1.5,
            },
            {
                "date": "2025-03-31",
                "component_family": "coupon_accrual",
                "component_anchored_interest_mil": 2.5,
            },
        ]
    )

    diagnostic = build_treasury_interest_expense_diagnostic(
        treasury,
        interest_detail,
        tier2_interest_candidate=candidate,
    )

    by_pool = diagnostic.set_index("official_pool")
    assert by_pool.loc["bill_discount", "official_interest_expense_mil"] == pytest.approx(2.0)
    assert by_pool.loc["coupon_accrual", "official_interest_expense_mil"] == pytest.approx(3.0)
    assert by_pool.loc["frn_interest", "official_interest_expense_mil"] == pytest.approx(4.0)
    assert by_pool.loc["tips_coupon_accrual", "official_interest_expense_mil"] == pytest.approx(5.0)
    assert by_pool.loc["tips_inflation_compensation", "official_interest_expense_mil"] == pytest.approx(6.0)
    assert by_pool.loc["bill_discount", "tdcest_candidate_component_mil"] == pytest.approx(1.5)
    assert by_pool.loc["coupon_accrual", "tdcest_candidate_component_mil"] == pytest.approx(2.5)
    assert pd.isna(by_pool.loc["tips_inflation_compensation", "replay_component_interest_mil"])
