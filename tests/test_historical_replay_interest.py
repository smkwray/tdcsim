from __future__ import annotations

import pandas as pd
import pytest

from historical_replay import _build_interest_proxy_alignment, _build_interest_proxy_alignment_summary
from historical_replay_interest import (
    DETAIL_COLUMNS,
    aggregate_stock_only_interest,
    align_stock_only_interest_proxy,
    build_bill_interest_flow_detail,
    build_bill_lot_conservation,
    build_fixed_coupon_interest_flow_detail,
    build_fixed_coupon_interest_reconciliation,
    build_fixed_coupon_principal_adjustments,
    build_frn_cusip_coverage,
    build_frn_daily_index_backcast_validation,
    build_frn_daily_index_path,
    build_frn_interest_flow_detail,
    build_frn_interest_reconciliation,
    build_frn_principal_reconciliation,
    build_interest_component_certification,
    build_interest_component_reconciliation,
    build_interest_scope_certification,
    build_nonbill_discount_premium_flow_detail,
    build_official_interest_scope_bridge,
    build_quarterly_interest_detail,
    build_sector_interest_allocation,
    build_sector_interest_totals,
    build_tips_inflation_compensation_flow_detail,
    build_tips_inflation_compensation_reconciliation,
    build_tips_coupon_accrual_flow_detail,
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
    assert bank["tdcest_point_clipped_to_stock_envelope"] == pytest.approx(1.2)
    assert bank["stock_only_marginal_min"] == pytest.approx(bank["feasible_min"])
    assert bank["stock_only_marginal_max"] == pytest.approx(bank["feasible_max"])
    assert bool(bank["tdcest_reference_intersects_stock_envelope"]) is True
    assert bool(bank["stock_only_within_tolerance"]) is False
    assert bool(bank["constrained_within_tolerance"]) is True
    assert bool(bank["within_tolerance"]) is True
    assert (
        bank["bounds_scope_status"]
        == "deprecated_non_joint_diagnostic_only_not_aggregate_reconciled_not_binding"
    )
    assert bool(bank["binding_downstream_bound"]) is False
    assert bool(bank["aggregate_reconciled_interest_bound"]) is False
    assert (
        bank["interest_proxy_claim_boundary"]
        == "quarter_end_stock_only_marginal_holder_envelope_not_event_ledger_or_joint_bound"
    )
    assert bank["method_tier"] == "single_holder_stock_envelope_diagnostic_projection"


def test_interest_alignment_summary_marks_stock_envelope_nonbinding(interest_detail: pd.DataFrame):
    tdc_panel = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "banks_plus_credit_union_tier2_candidate_interest_mil": 1.2,
            }
        ]
    )

    aligned = _build_interest_proxy_alignment(interest_detail, tdc_panel)
    summary = _build_interest_proxy_alignment_summary(aligned)

    assert "diagnostic_only_not_aggregate_reconciled_not_binding" in summary
    assert "deprecated, non-joint diagnostic projection" in summary
    assert "not aggregate-reconciled Treasury expense bounds" in summary
    assert "must not be used as a binding TDC-EST bound" in summary


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


def test_sector_interest_allocation_conserves_component_controls(
    quarterly_portfolio_snapshots: dict[str, pd.DataFrame],
    interest_detail: pd.DataFrame,
):
    controls = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component_id": "bill_discount",
                "official_interest_mil": 10.0,
                "model_interest_mil": 9.8,
                "certification_status": "certified_quarterly",
            },
            {
                "quarter": "2025Q1",
                "component_id": "coupon_accrual",
                "official_interest_mil": 20.0,
                "model_interest_mil": 19.9,
                "certification_status": "certified_quarterly",
            },
            {
                "quarter": "2025Q1",
                "component_id": "frn_interest",
                "official_interest_mil": 8.0,
                "model_interest_mil": 8.1,
                "certification_status": "certified_quarterly",
            },
            {
                "quarter": "2025Q1",
                "component_id": "tips_inflation_compensation",
                "official_interest_mil": 11.0,
                "model_interest_mil": 10.9,
                "certification_status": "certified_quarterly",
            },
            {
                "quarter": "2025Q1",
                "component_id": "tips_coupon_accrual",
                "official_interest_mil": 3.0,
                "model_interest_mil": 2.9,
                "certification_status": "candidate_timing_caveated",
            },
        ]
    )

    allocation = build_sector_interest_allocation(
        interest_detail,
        quarterly_portfolio_snapshots,
        controls,
    )

    by_component = allocation.groupby("component_id")["allocated_official_interest_mil"].sum()
    assert by_component.loc["bill_discount"] == pytest.approx(10.0)
    assert by_component.loc["coupon_accrual"] == pytest.approx(20.0)
    assert by_component.loc["frn_interest"] == pytest.approx(8.0)
    assert by_component.loc["tips_inflation_compensation"] == pytest.approx(11.0)
    assert by_component.loc["tips_coupon_accrual"] == pytest.approx(3.0)

    tips_inflation = allocation.loc[
        allocation["component_id"].eq("tips_inflation_compensation")
    ].iloc[0]
    assert tips_inflation["weight_source"] == "portfolio_tips_adjusted_principal_weight"
    assert tips_inflation["tdcsim_holder"] == f"Private:{PRIVATE_SUBBUCKET_DOMESTIC_NONBANK}"
    assert tips_inflation["tdc_sector"] == "Private"
    assert tips_inflation["allocation_status"] == "official_control_certified_component"

    totals = build_sector_interest_totals(allocation)
    private_core = totals.loc[
        totals["tdc_sector"].eq("Private")
        & totals["scope_id"].eq("certified_core_ex_tips_coupon_nonbill_dp")
    ].iloc[0]
    assert private_core["selected_allocated_interest_mil"] == pytest.approx(21.0)
    assert private_core["tdcsim_holder_count"] == 2

    private_extended = totals.loc[
        totals["tdc_sector"].eq("Private")
        & totals["scope_id"].eq("extended_with_timing_caveated_tips_coupon")
    ].iloc[0]
    assert private_extended["selected_allocated_interest_mil"] == pytest.approx(24.0)
    assert private_extended["tdcsim_holder_count"] == 2


def test_sector_interest_allocation_falls_back_to_stock_model_when_control_missing(
    quarterly_portfolio_snapshots: dict[str, pd.DataFrame],
    interest_detail: pd.DataFrame,
):
    allocation = build_sector_interest_allocation(
        interest_detail,
        quarterly_portfolio_snapshots,
        pd.DataFrame(),
    )

    coupon = allocation.loc[allocation["component_id"].eq("coupon_accrual")].iloc[0]
    assert coupon["aggregate_control_basis"] == "tdcsim_model_or_stock_component_pool"
    assert coupon["allocation_status"] == "tdcsim_stock_model_proxy_component"
    assert coupon["selected_allocated_interest_mil"] == pytest.approx(
        interest_detail.loc[interest_detail["component"].eq("fixed_coupon_accrual"), "interest_amount"].sum()
    )


def test_sector_interest_allocation_uses_bill_flow_prior_snapshot_and_unallocated():
    interest_detail = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "tdcsim_holder": "Banks",
                "component": "fixed_coupon_accrual",
                "interest_amount": 1.0,
                "excluded_from_default_canonical": False,
            }
        ]
    )
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2024Q4",
                "cusip": "BILL_A",
                "IssueDate": "2024-10-01",
                "MaturityDate": "2025-01-15",
                "SecurityType": "Fixed",
                "MaturityCategory": "bills",
                "FaceValue": 100.0,
                "tdcsim_holder": "Banks",
            }
        ]
    )
    bill_flow = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cusip": "BILL_A",
                "issue_date": "2024-10-01",
                "maturity_date": "2025-01-15",
                "modeled_interest_mil": 60.0,
            },
            {
                "quarter": "2025Q1",
                "cusip": "BILL_B",
                "issue_date": "2025-01-02",
                "maturity_date": "2025-03-15",
                "modeled_interest_mil": 40.0,
            },
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component_id": "bill_discount",
                "official_interest_mil": 100.0,
                "model_interest_mil": 100.0,
                "certification_status": "certified_quarterly",
            }
        ]
    )

    allocation = build_sector_interest_allocation(
        interest_detail,
        portfolio,
        controls,
        bill_interest_flow_detail=bill_flow,
    )

    bill_allocation = allocation.loc[allocation["component_id"].eq("bill_discount")].copy()
    by_sector = bill_allocation.set_index("tdc_sector")
    assert by_sector.loc["Banks", "selected_allocated_interest_mil"] == pytest.approx(60.0)
    assert by_sector.loc["Unallocated", "selected_allocated_interest_mil"] == pytest.approx(40.0)
    assert by_sector.loc["Banks", "weight_time_basis"].endswith("prior_quarter_snapshot")
    assert by_sector.loc["Banks", "attributed_weight_coverage_pct"] == pytest.approx(60.0)


def test_sector_interest_allocation_uses_frn_flow_not_par_weight():
    interest_detail = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "tdcsim_holder": "Banks",
                "component": "fixed_coupon_accrual",
                "interest_amount": 1.0,
                "excluded_from_default_canonical": False,
            }
        ]
    )
    portfolio = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cusip": "FRN_A",
                "IssueDate": "2024-01-31",
                "MaturityDate": "2026-01-31",
                "SecurityType": "FRN",
                "FaceValue": 100.0,
                "tdcsim_holder": "Banks",
            },
            {
                "quarter": "2025Q1",
                "cusip": "FRN_B",
                "IssueDate": "2024-01-31",
                "MaturityDate": "2026-01-31",
                "SecurityType": "FRN",
                "FaceValue": 100.0,
                "tdcsim_holder": "Foreign",
            },
        ]
    )
    frn_flow = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "cusip": "FRN_A",
                "issue_date": "2024-01-31",
                "maturity_date": "2026-01-31",
                "modeled_interest_mil": 10.0,
            },
            {
                "quarter": "2025Q1",
                "cusip": "FRN_B",
                "issue_date": "2024-01-31",
                "maturity_date": "2026-01-31",
                "modeled_interest_mil": 30.0,
            },
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component_id": "frn_interest",
                "official_interest_mil": 40.0,
                "model_interest_mil": 40.0,
                "certification_status": "certified_quarterly",
            }
        ]
    )

    allocation = build_sector_interest_allocation(
        interest_detail,
        portfolio,
        controls,
        frn_interest_flow_detail=frn_flow,
    )

    frn_allocation = allocation.loc[allocation["component_id"].eq("frn_interest")].copy()
    by_sector = frn_allocation.set_index("tdc_sector")
    assert by_sector.loc["Banks", "selected_allocated_interest_mil"] == pytest.approx(10.0)
    assert by_sector.loc["Foreign", "selected_allocated_interest_mil"] == pytest.approx(30.0)
    assert set(frn_allocation["weight_source"]) == {"frn_daily_interest_flow_holder_share_weight"}


def test_sector_interest_allocation_reports_source_basis_residual_as_unallocated_coverage():
    interest_detail = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "tdcsim_holder": "Banks",
                "component": "fixed_coupon_accrual",
                "interest_amount": 70.0,
                "excluded_from_default_canonical": False,
            },
            {
                "quarter": "2025Q1",
                "tdcsim_holder": "SourceBasisResidual",
                "component": "fixed_coupon_accrual",
                "interest_amount": 30.0,
                "excluded_from_default_canonical": True,
            },
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component_id": "coupon_accrual",
                "official_interest_mil": 100.0,
                "model_interest_mil": 100.0,
                "certification_status": "certified_quarterly",
            }
        ]
    )

    allocation = build_sector_interest_allocation(
        interest_detail,
        pd.DataFrame(columns=["quarter"]),
        controls,
    )

    by_sector = allocation.set_index("tdc_sector")
    assert by_sector.loc["Banks", "selected_allocated_interest_mil"] == pytest.approx(70.0)
    assert by_sector.loc["Unallocated", "selected_allocated_interest_mil"] == pytest.approx(30.0)
    assert by_sector.loc["Banks", "attributed_weight_coverage_pct"] == pytest.approx(70.0)
    assert by_sector.loc["Unallocated", "residual_weight_mil"] == pytest.approx(30.0)


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


def test_bill_interest_flow_accrues_auction_discount_by_overlap_days():
    lots = pd.DataFrame(
        [
            {
                "lot_id": "BILL1|2024-12-26|2024-12-31",
                "cusip": "BILL1",
                "auction_date": "2024-12-26",
                "issue_date": "2024-12-31",
                "maturity_date": "2025-03-31",
                "security_term": "13-Week",
                "price_per100": 99.0,
                "total_accepted": 100_000_000.0,
                "offering_amt": 90_000_000.0,
                "accepted_component_sum": 100_000_000.0,
            }
        ]
    )

    detail = build_bill_interest_flow_detail(lots, start_quarter="2024Q4", end_quarter="2025Q1")

    by_quarter = detail.set_index("quarter")
    assert by_quarter.loc["2024Q4", "overlap_days"] == 1
    assert by_quarter.loc["2025Q1", "overlap_days"] == 89
    assert by_quarter["modeled_interest_mil"].sum() == pytest.approx(1.0)
    assert by_quarter.loc["2025Q1", "modeled_interest_mil"] == pytest.approx(1.0 * 89.0 / 90.0)

    conservation = build_bill_lot_conservation(
        detail,
        start_quarter="2024Q4",
        end_quarter="2025Q1",
    )
    assert bool(conservation.loc[0, "conservation_pass"]) is True


def test_nonbill_discount_premium_flow_conserves_nominal_and_excludes_frn():
    lots = pd.DataFrame(
        [
            {
                "lot_id": "NOTE|2025-01-10|2025-01-15",
                "cusip": "912345AA1",
                "instrument_family": "Treasury Notes",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "first_interest_payment_date": "2025-07-15",
                "maturity_date": "2026-01-15",
                "called_date": "",
                "is_reopening": False,
                "price_source_tier": "price_per100",
                "clean_price_per100": 98.0,
                "par_mil": 100.0,
                "coupon_rate_decimal": 0.04,
                "reported_yield_pct": 5.0,
            },
            {
                "lot_id": "FRN|2025-01-10|2025-01-15",
                "cusip": "912345FRN",
                "instrument_family": "Treasury Floating Rate Notes (FRN)",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "first_interest_payment_date": "2025-07-15",
                "maturity_date": "2026-01-15",
                "called_date": "",
                "is_reopening": False,
                "price_source_tier": "price_per100",
                "clean_price_per100": 99.0,
                "par_mil": 100.0,
                "coupon_rate_decimal": 0.0,
                "reported_yield_pct": 5.0,
            },
        ]
    )

    detail = build_nonbill_discount_premium_flow_detail(lots, start_quarter="2025Q1", end_quarter="2026Q1")

    assert set(detail["lot_id"]) == {"NOTE|2025-01-10|2025-01-15"}
    assert detail["quarter_amortization_mil"].sum() == pytest.approx(2.0, abs=1e-10)
    assert detail["terminal_residual_per100"].max() < 1e-8
    assert set(detail["dp_sign"]) == {"discount"}


def test_nonbill_discount_premium_flow_keeps_tips_real_unit_amortization_unscaled():
    lots = pd.DataFrame(
        [
            {
                "lot_id": "TIPS|2025-01-10|2025-01-15",
                "cusip": "912345TI1",
                "instrument_family": "Treasury Inflation Protected Securities (TIPS)",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "first_interest_payment_date": "2025-07-15",
                "maturity_date": "2026-01-15",
                "called_date": "",
                "is_reopening": False,
                "price_source_tier": "unadj_price",
                "clean_price_per100": 101.0,
                "par_mil": 200.0,
                "coupon_rate_decimal": 0.01,
                "reported_yield_pct": 0.5,
            },
        ]
    )

    detail = build_nonbill_discount_premium_flow_detail(lots, start_quarter="2025Q1", end_quarter="2026Q1")

    assert detail["quarter_amortization_mil"].sum() == pytest.approx(-2.0, abs=1e-10)
    assert detail["source_identity_status"].unique().tolist() == [
        "tips_real_unit_effective_interest_source_ledger"
    ]


def test_interest_certification_excludes_nonbill_discount_premium_and_tips_coupon_from_scope():
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED DISCOUNT",
                "expense_type_desc": "Treasury Notes",
                "month_expense_amt": 2_000_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED PREMIUM",
                "expense_type_desc": "Treasury Floating Rate Notes (FRN)",
                "month_expense_amt": -500_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Inflation Protected Securities (TIPS)",
                "month_expense_amt": 100_000.0,
            },
        ]
    )
    reconciliation = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component": "bill_discount",
                "selected_model_interest_mil": 0.0,
                "source_coverage_pct": 100.0,
                "canonical_status": "canonical_aggregate_bill_discount",
                "selected_model_basis": "auction_lot_discount_flow",
            },
        ]
    )
    nonbill = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "instrument_family": "Treasury Notes",
                "quarter_amortization_mil": 2.0,
                "terminal_residual_per100": 0.0,
                "source_identity_status": "nominal_effective_interest_source_ledger",
                "price_source_tier": "price_per100",
                "lot_id": "NOTE",
                "model_version": "nonbill_effective_interest_discount_premium_v1",
            },
        ]
    )

    component = build_interest_component_certification(treasury, reconciliation, nonbill)
    scope = build_interest_scope_certification(component)

    by_component = component.set_index("component_id")
    assert bool(by_component.loc["nominal_nonbill_discount_premium", "included_in_scope"]) is False
    assert by_component.loc["nominal_nonbill_discount_premium", "certification_status"] == "excluded_selected_scope"
    assert (
        by_component.loc["nominal_nonbill_discount_premium", "failure_code"]
        == "PREMATURITY_REDEMPTION_ACCOUNTING_UNRESOLVED"
    )
    assert bool(by_component.loc["frn_discount_premium", "included_in_scope"]) is False
    assert by_component.loc["frn_discount_premium", "certification_status"] == "excluded_selected_scope"
    assert bool(by_component.loc["tips_coupon_accrual", "included_in_scope"]) is False
    assert by_component.loc["tips_coupon_accrual", "certification_status"] == "candidate_timing_caveated"
    assert "nominal_nonbill_discount_premium" not in scope.loc[0, "expected_components"]
    assert scope.loc[0, "official_excluded_amount_mil"] == pytest.approx(1.6)


def test_treasury_interest_expense_diagnostic_prefers_bill_flow_when_supplied(
    interest_detail: pd.DataFrame,
):
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED DISCOUNT",
                "expense_type_desc": "Treasury Bills",
                "month_expense_amt": 2_000_000.0,
            }
        ]
    )
    flow = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component": "bill_discount",
                "modeled_interest_mil": 1.95,
            }
        ]
    )

    diagnostic = build_treasury_interest_expense_diagnostic(
        treasury,
        interest_detail,
        bill_interest_flow_detail=flow,
    )

    bill = diagnostic.set_index("official_pool").loc["bill_discount"]
    assert bill["replay_component_interest_mil"] == pytest.approx(1.95)
    assert bill["replay_minus_official_mil"] == pytest.approx(-0.05)


def test_interest_component_reconciliation_keeps_stock_and_selected_bill_flow(
    interest_detail: pd.DataFrame,
):
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED DISCOUNT",
                "expense_type_desc": "Treasury Bills",
                "month_expense_amt": 2_000_000.0,
            }
        ]
    )
    flow = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "component": "bill_discount",
                "modeled_interest_mil": 1.95,
            }
        ]
    )

    reconciliation = build_interest_component_reconciliation(
        treasury,
        interest_detail,
        bill_interest_flow_detail=flow,
    )

    bill = reconciliation.set_index("component").loc["bill_discount"]
    assert bill["stock_only_interest_mil"] != pytest.approx(1.95)
    assert bill["flow_interest_mil"] == pytest.approx(1.95)
    assert bill["selected_model_interest_mil"] == pytest.approx(1.95)
    assert bill["selected_model_basis"] == "auction_lot_discount_flow"
    assert bill["scope_id"] == "marketable_coupon_like_ex_tips_inflation"
    assert bill["canonical_status"] == "canonical_aggregate_bill_discount"


def test_fixed_coupon_flow_keeps_prior_only_maturing_cusip_and_closes_residual_identity():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "quarter": "2024Q4",
                "cusip": "912345AA1",
                "security_class": "Notes",
                "outstanding_mil": 100.0,
                "coupon_rate_decimal": 0.06,
                "maturity_date": "2025-01-15",
                "interest_pay_date_1": "01/15",
                "interest_pay_date_2": "07/15",
            }
        ]
    )

    detail = build_fixed_coupon_interest_flow_detail(
        stocks,
        pd.DataFrame(),
        start_quarter="2025Q1",
        end_quarter="2025Q1",
    )

    assert len(detail) == 1
    row = detail.iloc[0]
    assert row["record_date"] == "2025-01-31"
    assert row["opening_par_mil"] == pytest.approx(100.0)
    assert row["closing_par_mil"] == pytest.approx(0.0)
    assert row["known_redemption_mil"] == pytest.approx(100.0)
    assert row["unexplained_principal_change_mil"] == pytest.approx(0.0)
    assert row["modeled_interest_mil"] == pytest.approx(100.0 * 0.06 * 15.0 / 184.0 * 0.5)


def test_fixed_coupon_flow_uses_dated_issuance_and_reports_residual_bounds():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "quarter": "2024Q4",
                "cusip": "912345AA1",
                "security_class": "Notes",
                "outstanding_mil": 100.0,
                "coupon_rate_decimal": 0.06,
                "maturity_date": "2029-01-15",
                "interest_pay_date_1": "01/15",
                "interest_pay_date_2": "07/15",
            },
            {
                "record_date": "2025-01-31",
                "quarter": "2025Q1",
                "cusip": "912345AA1",
                "security_class": "Notes",
                "outstanding_mil": 160.0,
                "coupon_rate_decimal": 0.06,
                "maturity_date": "2029-01-15",
                "interest_pay_date_1": "01/15",
                "interest_pay_date_2": "07/15",
            },
        ]
    )
    auctions = pd.DataFrame(
        [
            {
                "event_id": "912345AA1|2025-01-10|2025-01-15",
                "cusip": "912345AA1",
                "security_class": "Notes",
                "auction_date": "2025-01-10",
                "issue_date": "2025-01-15",
                "maturity_date": "2029-01-15",
                "called_date": "",
                "total_accepted_mil": 50.0,
            }
        ]
    )

    detail = build_fixed_coupon_interest_flow_detail(
        stocks,
        auctions,
        start_quarter="2025Q1",
        end_quarter="2025Q1",
    )

    row = detail.loc[detail["record_date"].eq("2025-01-31")].iloc[0]
    assert row["known_issuance_mil"] == pytest.approx(50.0)
    assert row["unexplained_principal_change_mil"] == pytest.approx(10.0)
    assert row["residual_interest_low_mil"] <= row["residual_interest_mid_mil"] <= row["residual_interest_high_mil"]
    adjustments = build_fixed_coupon_principal_adjustments(detail)
    assert set(adjustments["event_type"]) == {"issuance", "residual"}


def test_reconciliations_can_select_fixed_coupon_flow():
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Treasury Notes",
                "month_expense_amt": 4_000_000.0,
            },
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Treasury Bonds",
                "month_expense_amt": 6_000_000.0,
            },
        ]
    )
    fixed_flow = pd.DataFrame(
        [
            {
                "quarter": "2025Q1",
                "security_class": "Notes",
                "modeled_interest_mil": 4.1,
                "residual_interest_low_mil": 0.0,
                "residual_interest_high_mil": 0.1,
            },
            {
                "quarter": "2025Q1",
                "security_class": "Bonds",
                "modeled_interest_mil": 5.9,
                "residual_interest_low_mil": 0.0,
                "residual_interest_high_mil": 0.1,
            },
        ]
    )

    diagnostic = build_treasury_interest_expense_diagnostic(
        treasury,
        pd.DataFrame(columns=DETAIL_COLUMNS),
        fixed_coupon_flow_detail=fixed_flow,
    )
    assert diagnostic.set_index("official_pool").loc["coupon_accrual", "replay_component_interest_mil"] == pytest.approx(10.0)

    component = build_interest_component_reconciliation(
        treasury,
        pd.DataFrame(columns=DETAIL_COLUMNS),
        fixed_coupon_flow_detail=fixed_flow,
    ).set_index("component").loc["coupon_accrual"]
    assert component["selected_model_interest_mil"] == pytest.approx(10.0)
    assert component["selected_model_basis"] == "mspd_auction_event_coupon_flow"

    fixed = build_fixed_coupon_interest_reconciliation(treasury, fixed_flow)
    combined = fixed.set_index(["quarter", "security_class"]).loc[("2025Q1", "Notes+Bonds")]
    assert combined["official_interest_mil"] == pytest.approx(10.0)
    assert combined["modeled_interest_mil"] == pytest.approx(10.0)


def test_frn_daily_index_path_applies_original_globally_and_reopening_by_cusip():
    benchmark = pd.DataFrame(
        [
            {"auction_date": "2024-10-21", "effective_date": "2024-10-22", "benchmark_index_pct": 4.30, "term_days": 91},
            {"auction_date": "2024-10-28", "effective_date": "2024-10-29", "benchmark_index_pct": 4.40, "term_days": 91},
            {"auction_date": "2024-11-18", "effective_date": "2024-11-19", "benchmark_index_pct": 4.10, "term_days": 91},
        ]
    )
    lots = pd.DataFrame(
        [
            {
                "lot_id": "AAA|2024-01-29|2024-01-31",
                "cusip": "AAA",
                "auction_date": "2024-01-29",
                "issue_date": "2024-01-31",
                "maturity_date": "2026-01-31",
                "total_accepted_mil": 10_000.0,
                "spread_pct": 0.20,
                "is_reopening": False,
                "frn_index_determination_date": "2024-01-22",
                "frn_index_determination_rate": 5.0,
            },
            {
                "lot_id": "BBB|2024-10-29|2024-10-31",
                "cusip": "BBB",
                "auction_date": "2024-10-29",
                "issue_date": "2024-10-31",
                "maturity_date": "2026-10-31",
                "total_accepted_mil": 15_000.0,
                "spread_pct": 0.205,
                "is_reopening": False,
                "frn_index_determination_date": "2024-10-21",
                "frn_index_determination_rate": 4.51,
            },
            {
                "lot_id": "BBB|2024-11-26|2024-11-29",
                "cusip": "BBB",
                "auction_date": "2024-11-26",
                "issue_date": "2024-11-29",
                "maturity_date": "2026-10-31",
                "total_accepted_mil": 2_000.0,
                "spread_pct": 0.205,
                "is_reopening": True,
                "frn_index_determination_date": "2024-11-18",
                "frn_index_determination_rate": 4.42,
            },
        ]
    )

    path = build_frn_daily_index_path(benchmark, lots, start_quarter="2024Q4", end_quarter="2024Q4")
    keyed = path.set_index(["cusip", "accrual_date"])
    original_override = round(4.51 / (1.0 - (4.51 / 100.0) * 91.0 / 360.0), 9)
    reopening_override = round(4.42 / (1.0 - (4.42 / 100.0) * 91.0 / 360.0), 9)

    assert keyed.loc[("AAA", pd.Timestamp("2024-10-29")), "daily_index_pct"] == pytest.approx(original_override)
    assert keyed.loc[("AAA", pd.Timestamp("2024-10-29")), "index_source"] == (
        "frn_original_issue_index_determination_lockout"
    )
    assert keyed.loc[("BBB", pd.Timestamp("2024-10-31")), "daily_index_pct"] == pytest.approx(original_override)
    assert keyed.loc[("BBB", pd.Timestamp("2024-11-27")), "daily_index_pct"] == pytest.approx(reopening_override)
    assert keyed.loc[("AAA", pd.Timestamp("2024-11-27")), "daily_index_pct"] == pytest.approx(4.10)


def test_frn_daily_index_validation_matches_observed_rows():
    daily_path = pd.DataFrame(
        [
            {
                "cusip": "AAA",
                "accrual_date": "2024-10-31",
                "daily_index_pct": 4.562008,
                "index_source": "13_week_bill_high_discount_rate",
                "model_version": "test",
            }
        ]
    )
    observed = pd.DataFrame(
        [
            {
                "cusip": "AAA",
                "start_of_accrual_period": "2024-10-31",
                "daily_index": 4.562008,
                "spread": 0.205,
                "daily_int_accrual_rate": 4.767008,
                "daily_accrued_int_per100": 4.767008 / 360.0,
            }
        ]
    )

    validation = build_frn_daily_index_backcast_validation(daily_path, observed)

    assert bool(validation.loc[0, "validation_pass"]) is True
    assert validation.loc[0, "daily_accrued_int_per100_diff"] == pytest.approx(0.0)


def test_frn_interest_flow_detail_computes_actual_360_interest():
    lots = pd.DataFrame(
        [
            {
                "lot_id": "AAA|2024-10-29|2024-10-31",
                "cusip": "AAA",
                "auction_date": "2024-10-29",
                "issue_date": "2024-10-31",
                "maturity_date": "2024-11-01",
                "total_accepted_mil": 3_600.0,
                "spread_pct": 0.6,
                "is_reopening": False,
            }
        ]
    )
    daily_path = pd.DataFrame(
        [
            {
                "cusip": "AAA",
                "accrual_date": "2024-10-31",
                "daily_index_pct": 3.0,
                "index_source": "13_week_bill_high_discount_rate",
                "model_version": "test",
            }
        ]
    )

    detail = build_frn_interest_flow_detail(lots, daily_path)

    assert len(detail) == 1
    assert detail.loc[0, "daily_int_accrual_rate_pct"] == pytest.approx(3.6)
    assert detail.loc[0, "daily_accrued_int_per100"] == pytest.approx(3.6 / 360.0)
    assert detail.loc[0, "modeled_interest_mil"] == pytest.approx(0.36)


def test_frn_reconciliations_and_coverage_select_flow():
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Treasury Floating Rate Notes (FRN)",
                "month_expense_amt": 4_000_000.0,
            }
        ]
    )
    flow = pd.DataFrame([{"quarter": "2025Q1", "cusip": "AAA", "modeled_interest_mil": 4.01}])

    diagnostic = build_treasury_interest_expense_diagnostic(
        treasury,
        pd.DataFrame(columns=DETAIL_COLUMNS),
        frn_interest_flow_detail=flow,
    )
    assert diagnostic.set_index("official_pool").loc["frn_interest", "replay_component_interest_mil"] == pytest.approx(4.01)

    component = build_interest_component_reconciliation(
        treasury,
        pd.DataFrame(columns=DETAIL_COLUMNS),
        frn_interest_flow_detail=flow,
    ).set_index("component").loc["frn_interest"]
    assert component["selected_model_interest_mil"] == pytest.approx(4.01)
    assert component["selected_model_basis"] == "frn_auction_lot_benchmark_accrual_flow"

    recon = build_frn_interest_reconciliation(treasury, flow)
    assert recon.loc[0, "gap_mil"] == pytest.approx(0.01)

    principal = build_frn_principal_reconciliation(
        pd.DataFrame(
            [{"record_date": "2025-03-31", "quarter": "2025Q1", "cusip": "AAA", "control_outstanding_mil": 100.0}]
        ),
        pd.DataFrame(
            [{"cusip": "AAA", "issue_date": "2024-01-31", "maturity_date": "2026-01-31", "total_accepted_mil": 100.0}]
        ),
    )
    assert principal.loc[0, "difference_mil"] == pytest.approx(0.0)

    coverage = build_frn_cusip_coverage(
        pd.DataFrame(
            [{"cusip": "AAA", "issue_date": "2024-01-31", "maturity_date": "2026-01-31", "total_accepted_mil": 100.0}]
        ),
        pd.DataFrame([{"cusip": "AAA", "accrual_date": "2025-03-31"}]),
        pd.DataFrame([{"cusip": "AAA", "start_of_accrual_period": "2025-03-31"}]),
        pd.DataFrame([{"cusip": "AAA", "record_date": "2025-03-31"}]),
    )
    assert coverage.loc[0, "coverage_status"] == "auction_lot_modeled_daily_observed_daily_mspd_control"


def test_official_interest_scope_bridge_maps_included_and_excluded_rows():
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-12-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED DISCOUNT",
                "expense_type_desc": "Treasury Bills",
                "month_expense_amt": 2_000_000.0,
            },
            {
                "record_date": "2025-12-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Treasury Notes",
                "month_expense_amt": 3_000_000.0,
            },
            {
                "record_date": "2025-12-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Int. Expense Inflation Compensation (TIPS)",
                "month_expense_amt": 4_000_000.0,
            },
            {
                "record_date": "2025-12-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "AMORTIZED PREMIUM",
                "expense_type_desc": "Treasury Notes",
                "month_expense_amt": -5_000_000.0,
            },
            {
                "record_date": "2025-12-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "SAVINGS BONDS",
                "expense_type_desc": "Series I",
                "month_expense_amt": 6_000_000.0,
            },
            {
                "record_date": "2025-12-31",
                "expense_catg_desc": "INTEREST EXPENSE ON GOVT ACCOUNT SERIES",
                "expense_group_desc": "ACCRUAL BASIS GAS EXPENSE",
                "expense_type_desc": "Zero Coupon Bonds Interest Expense",
                "month_expense_amt": 7_000_000.0,
            },
        ]
    )

    bridge = build_official_interest_scope_bridge(treasury)
    by_type = bridge.set_index("expense_type_desc")

    assert by_type.loc["Treasury Bills", "component"] == "bill_discount"
    assert bool(by_type.loc["Treasury Bills", "include_in_coupon_like_scope"]) is True
    assert bool(by_type.loc["Treasury Bills", "include_in_full_marketable_accrual_scope"]) is True
    assert by_type.loc["Treasury Notes", "component"].iloc[0] == "coupon_accrual"
    assert bool(by_type.loc["Int. Expense Inflation Compensation (TIPS)", "include_in_coupon_like_scope"]) is False
    assert bool(by_type.loc["Int. Expense Inflation Compensation (TIPS)", "include_in_full_marketable_accrual_scope"]) is True
    assert "excluded_from_coupon_like_scope" in by_type.loc[
        "Int. Expense Inflation Compensation (TIPS)",
        "exclusion_reason",
    ]
    premium_note = bridge[
        bridge["expense_group_desc"].eq("AMORTIZED PREMIUM")
        & bridge["expense_type_desc"].eq("Treasury Notes")
    ].iloc[0]
    assert premium_note["component"] == "nonbill_discount_premium"
    assert bool(premium_note["include_in_coupon_like_scope"]) is False
    assert bool(premium_note["include_in_full_marketable_accrual_scope"]) is True
    assert by_type.loc["Series I", "exclusion_reason"] == "savings_bonds_outside_marketable_replay"
    assert (
        by_type.loc["Zero Coupon Bonds Interest Expense", "exclusion_reason"]
        == "government_account_series_outside_marketable_replay"
    )


def test_official_interest_scope_bridge_covers_live_source_categories():
    treasury = pd.read_csv(
        "data/historical_replay/imported/tdcest/treasury__interest_expense.csv",
        low_memory=False,
    )

    bridge = build_official_interest_scope_bridge(treasury)

    assert len(bridge) == len(treasury)
    assert not bridge["scope_id"].eq("unmapped").any()
    assert not bridge["component"].isna().any()


def test_tips_inflation_compensation_delta_flow_selects_official_component():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "quarter": "2024Q4",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 120.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-01-31",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 125.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-02-28",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 123.0,
                "redeemed_amt_mil": 0.0,
            },
        ]
    )
    treasury = pd.DataFrame(
        [
            {
                "record_date": "2025-03-31",
                "expense_catg_desc": "INTEREST EXPENSE ON PUBLIC ISSUES",
                "expense_group_desc": "ACCRUED INTEREST EXPENSE",
                "expense_type_desc": "Int. Expense Inflation Compensation (TIPS)",
                "month_expense_amt": 3_000_000.0,
            }
        ]
    )

    flow = build_tips_inflation_compensation_flow_detail(
        stocks,
        start_quarter="2025Q1",
        end_quarter="2025Q1",
    )

    assert flow["modeled_inflation_compensation_mil"].tolist() == pytest.approx([5.0, -2.0])
    assert not flow["is_first_observed_cohort_month"].any()

    diagnostic = build_treasury_interest_expense_diagnostic(
        treasury,
        pd.DataFrame(columns=DETAIL_COLUMNS),
        tips_inflation_flow_detail=flow,
    )
    assert diagnostic.set_index("official_pool").loc[
        "tips_inflation_compensation",
        "replay_component_interest_mil",
    ] == pytest.approx(3.0)

    component = build_interest_component_reconciliation(
        treasury,
        pd.DataFrame(columns=DETAIL_COLUMNS),
        tips_inflation_flow_detail=flow,
    ).set_index("component").loc["tips_inflation_compensation"]
    assert component["selected_model_interest_mil"] == pytest.approx(3.0)
    assert component["selected_model_basis"] == "mspd_cusip_inflation_adjustment_event_flow"
    assert component["scope_id"] == "full_marketable_accrual"
    assert component["canonical_status"] == (
        "candidate_aggregate_tips_inflation_compensation_event_flow"
    )

    reconciliation = build_tips_inflation_compensation_reconciliation(treasury, flow)
    assert reconciliation.loc[0, "official_interest_mil"] == pytest.approx(3.0)
    assert reconciliation.loc[0, "modeled_interest_mil"] == pytest.approx(3.0)
    assert reconciliation.loc[0, "gap_mil"] == pytest.approx(0.0)


def test_tips_inflation_compensation_adds_redeemed_adjustment():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "quarter": "2024Q4",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 100.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-01-31",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 105.0,
                "redeemed_amt_mil": -10.0,
            },
        ]
    )

    flow = build_tips_inflation_compensation_flow_detail(
        stocks,
        start_quarter="2025Q1",
        end_quarter="2025Q1",
    )

    expected_redemption_adjustment = 10.0 * 105.0 / 990.0
    assert flow.loc[0, "stock_delta_inflation_adjustment_mil"] == pytest.approx(5.0)
    assert flow.loc[0, "redeemed_increment_mil"] == pytest.approx(10.0)
    assert flow.loc[0, "redemption_adjustment_mil"] == pytest.approx(expected_redemption_adjustment)
    assert flow.loc[0, "modeled_inflation_compensation_mil"] == pytest.approx(
        5.0 + expected_redemption_adjustment
    )


def test_tips_inflation_compensation_subtracts_issue_adjustment_and_adds_maturity_floor():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "quarter": "2024Q4",
                "cohort_id": "912810AA1|2024-12-15|2025-02-15",
                "cusip": "912810AA1",
                "issue_date": "2024-12-15",
                "maturity_date": "2025-02-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 100.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-01-31",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2024-12-15|2025-02-15",
                "cusip": "912810AA1",
                "issue_date": "2024-12-15",
                "maturity_date": "2025-02-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 110.0,
                "redeemed_amt_mil": 0.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "cusip": "912810AA1",
                "issue_date": "2025-01-15",
                "maturity_date": "2025-02-15",
                "total_accepted_mil": 100.0,
                "index_ratio_on_issue_date": 1.08,
                "ref_cpi_on_dated_date": 200.0,
                "ref_cpi_on_issue_date": 216.0,
            }
        ]
    )
    cpi = pd.DataFrame([{"index_date": "2025-02-15", "ref_cpi": 230.0}])

    flow = build_tips_inflation_compensation_flow_detail(
        stocks,
        start_quarter="2025Q1",
        end_quarter="2025Q1",
        tips_auction_events=events,
        tips_cpi_reference_path=cpi,
    )

    stock = flow.loc[flow["event_type"].eq("mspd_cusip_stock_flow")].iloc[0]
    maturity = flow.loc[flow["event_type"].eq("synthetic_maturity_floor_event")].iloc[0]
    assert stock["stock_delta_inflation_adjustment_mil"] == pytest.approx(10.0)
    assert stock["issue_adjustment_mil"] == pytest.approx(8.0)
    assert stock["modeled_inflation_compensation_mil"] == pytest.approx(2.0)
    assert maturity["maturity_floor_adjustment_mil"] == pytest.approx(150.0)
    assert maturity["modeled_inflation_compensation_mil"] == pytest.approx(40.0)


def test_tips_inflation_compensation_aggregates_cusip_row_splits():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2025-01-31",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 100.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-02-28",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 600.0,
                "inflation_adjustment_mil": 61.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-02-28",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2021-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2021-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 400.0,
                "inflation_adjustment_mil": 40.0,
                "redeemed_amt_mil": 0.0,
            },
        ]
    )

    flow = build_tips_inflation_compensation_flow_detail(stocks, start_quarter="2025Q1", end_quarter="2025Q1")

    assert len(flow) == 2
    assert flow.loc[flow["record_date"].eq("2025-02-28"), "modeled_inflation_compensation_mil"].iloc[0] == pytest.approx(1.0)


def test_tips_inflation_compensation_rejects_missing_months():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2024-12-31",
                "quarter": "2024Q4",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 100.0,
                "redeemed_amt_mil": 0.0,
            },
            {
                "record_date": "2025-02-28",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-01-15|2030-01-15",
                "cusip": "912810AA1",
                "issue_date": "2020-01-15",
                "maturity_date": "2030-01-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 105.0,
                "redeemed_amt_mil": 0.0,
            },
        ]
    )

    with pytest.raises(ValueError, match="non-monthly gaps"):
        build_tips_inflation_compensation_flow_detail(stocks)


def test_tips_coupon_accrual_flow_stops_at_maturity_and_subtracts_issue_accrued():
    stocks = pd.DataFrame(
        [
            {
                "record_date": "2025-01-31",
                "quarter": "2025Q1",
                "cohort_id": "912810AA1|2020-02-15|2025-02-15",
                "cusip": "912810AA1",
                "issue_date": "2020-02-15",
                "maturity_date": "2025-02-15",
                "issued_amt_mil": 1000.0,
                "inflation_adjustment_mil": 100.0,
                "redeemed_amt_mil": 0.0,
            },
        ]
    )
    events = pd.DataFrame(
        [
            {
                "cusip": "912810AA1",
                "issue_date": "2020-02-15",
                "maturity_date": "2025-02-15",
                "total_accepted_mil": 1000.0,
                "coupon_rate_decimal": 0.02,
                "ref_cpi_on_dated_date": 200.0,
                "adj_accrued_int_per1000": 0.0,
            },
            {
                "cusip": "912810AA1",
                "issue_date": "2025-01-15",
                "maturity_date": "2025-02-15",
                "total_accepted_mil": 100.0,
                "coupon_rate_decimal": 0.02,
                "ref_cpi_on_dated_date": 200.0,
                "adj_accrued_int_per1000": 5.0,
            },
        ]
    )
    cpi = pd.DataFrame(
        [
            {"index_date": "2025-01-01", "ref_cpi": 220.0},
            {"index_date": "2025-01-15", "ref_cpi": 220.0},
            {"index_date": "2025-02-15", "ref_cpi": 220.0},
            {"index_date": "2025-03-31", "ref_cpi": 220.0},
        ]
    )

    flow = build_tips_coupon_accrual_flow_detail(
        stocks,
        events,
        cpi,
        start_quarter="2025Q1",
        end_quarter="2025Q2",
    )

    q1 = flow.set_index("quarter").loc["2025Q1"]
    assert q1["cash_coupon_mil"] == pytest.approx(11.0)
    assert q1["issue_accrued_interest_mil"] == pytest.approx(0.5)
    assert q1["accrued_interest_close_mil"] == pytest.approx(0.0)
    assert q1["modeled_interest_mil"] == pytest.approx(10.5)
    assert flow.set_index("quarter").loc["2025Q2", "modeled_interest_mil"] == pytest.approx(0.0)
