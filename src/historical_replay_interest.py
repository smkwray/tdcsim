"""Quarterly stock-only interest detail helpers for historical replay snapshots."""

from __future__ import annotations

import calendar
from collections.abc import Mapping

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay, DateOffset

from tdc_shared import DAYS_PER_YEAR_ACTUAL, TGA_FLOOR_TOLERANCE


DETAIL_COLUMNS = [
    "quarter",
    "cusip/cohort_id",
    "native_sector",
    "broad_holder_class",
    "tdcsim_holder",
    "security_type",
    "component",
    "issue_date",
    "maturity_date",
    "exposure_start",
    "exposure_end",
    "days_held",
    "exposure_fraction",
    "face_begin",
    "face_end",
    "exposure_base",
    "coupon_rate_decimal",
    "issue_price_ratio",
    "issue_yield_decimal",
    "fixed_spread",
    "index_ratio_start",
    "index_ratio_end",
    "ref_cpi_start",
    "ref_cpi_end",
    "derivation_status",
    "source_status",
    "interest_amount",
    "excluded_from_default_canonical",
]

TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS = [
    "quarter",
    "official_pool",
    "official_interest_expense_mil",
    "official_row_count",
    "replay_component_interest_mil",
    "tdcest_candidate_component_mil",
    "replay_minus_official_mil",
    "tdcest_candidate_minus_official_mil",
    "diagnostic_status",
]

BILL_INTEREST_FLOW_DETAIL_COLUMNS = [
    "quarter",
    "component",
    "lot_id",
    "cusip",
    "auction_date",
    "issue_date",
    "maturity_date",
    "security_term",
    "par_basis",
    "total_accepted_mil",
    "offering_amt_mil",
    "accepted_component_sum_mil",
    "accepted_component_residual_mil",
    "price_per100",
    "issue_proceeds_mil",
    "lifetime_discount_mil",
    "accrual_start",
    "accrual_end_exclusive",
    "term_days",
    "overlap_days",
    "modeled_interest_mil",
    "source_coverage_status",
    "model_version",
    "canonical_status",
]

BILL_LOT_CONSERVATION_COLUMNS = [
    "lot_id",
    "lifetime_discount_mil",
    "allocated_interest_mil",
    "pre_window_unallocated_mil",
    "post_window_unallocated_mil",
    "conservation_residual_mil",
    "conservation_pass",
]

INTEREST_COMPONENT_RECONCILIATION_COLUMNS = [
    "quarter",
    "scope_id",
    "component",
    "official_interest_mil",
    "stock_only_interest_mil",
    "flow_interest_mil",
    "selected_model_interest_mil",
    "selected_model_basis",
    "gap_mil",
    "gap_pct",
    "source_coverage_pct",
    "canonical_status",
    "exclusion_reason",
]

OFFICIAL_INTEREST_SCOPE_BRIDGE_COLUMNS = [
    "quarter",
    "record_date",
    "expense_catg_desc",
    "expense_group_desc",
    "expense_type_desc",
    "month_expense_amt",
    "scope_id",
    "component",
    "include_in_coupon_like_scope",
    "include_in_full_marketable_accrual_scope",
    "exclusion_reason",
]

FIXED_COUPON_MONTHLY_DETAIL_COLUMNS = [
    "record_date",
    "quarter",
    "cusip",
    "security_class",
    "opening_par_mil",
    "closing_par_mil",
    "coupon_rate_decimal",
    "maturity_date",
    "interest_pay_date_1",
    "interest_pay_date_2",
    "full_month_coupon_factor",
    "known_issuance_mil",
    "known_redemption_mil",
    "issuance_interest_mil",
    "redemption_interest_mil",
    "unexplained_principal_change_mil",
    "residual_timing_assumption",
    "residual_interest_low_mil",
    "residual_interest_mid_mil",
    "residual_interest_high_mil",
    "modeled_interest_mil",
    "source_coverage_status",
    "model_version",
    "canonical_status",
]

FIXED_COUPON_PRINCIPAL_ADJUSTMENT_COLUMNS = [
    "record_date",
    "quarter",
    "cusip",
    "security_class",
    "event_type",
    "event_date",
    "principal_mil",
    "source",
    "model_version",
]

FIXED_COUPON_INTEREST_RECONCILIATION_COLUMNS = [
    "quarter",
    "security_class",
    "official_interest_mil",
    "modeled_interest_mil",
    "gap_mil",
    "abs_gap_mil",
    "gap_pct",
    "abs_gap_pct",
    "residual_timing_low_mil",
    "residual_timing_high_mil",
    "residual_timing_envelope_width_mil",
    "source_coverage_pct",
    "model_version",
    "canonical_status",
]

FRN_DAILY_INDEX_PATH_COLUMNS = [
    "cusip",
    "accrual_date",
    "daily_index_pct",
    "index_source",
    "model_version",
]

FRN_DAILY_INDEX_VALIDATION_COLUMNS = [
    "cusip",
    "accrual_date",
    "observed_daily_index_pct",
    "modeled_daily_index_pct",
    "index_diff_pct",
    "observed_spread_pct",
    "modeled_spread_pct",
    "spread_diff_pct",
    "observed_daily_int_accrual_rate_pct",
    "modeled_daily_int_accrual_rate_pct",
    "accrual_rate_diff_pct",
    "observed_daily_accrued_int_per100",
    "modeled_daily_accrued_int_per100",
    "daily_accrued_int_per100_diff",
    "validation_pass",
]

FRN_INTEREST_DAILY_DETAIL_COLUMNS = [
    "quarter",
    "accrual_date",
    "lot_id",
    "cusip",
    "auction_date",
    "issue_date",
    "maturity_date",
    "is_reopening",
    "par_mil",
    "daily_index_pct",
    "spread_pct",
    "daily_int_accrual_rate_pct",
    "daily_accrued_int_per100",
    "modeled_interest_mil",
    "source_coverage_status",
    "model_version",
    "canonical_status",
]

FRN_PRINCIPAL_RECONCILIATION_COLUMNS = [
    "record_date",
    "quarter",
    "cusip",
    "mspd_control_outstanding_mil",
    "auction_lot_outstanding_mil",
    "difference_mil",
    "abs_difference_mil",
    "difference_pct",
    "source_coverage_status",
    "model_version",
]

FRN_CUSIP_COVERAGE_COLUMNS = [
    "cusip",
    "first_issue_date",
    "maturity_date",
    "auction_lot_count",
    "auction_lot_par_mil",
    "modeled_daily_rows",
    "observed_daily_rows",
    "mspd_control_rows",
    "coverage_status",
    "model_version",
]

FRN_INTEREST_RECONCILIATION_COLUMNS = [
    "quarter",
    "official_interest_mil",
    "modeled_interest_mil",
    "gap_mil",
    "abs_gap_mil",
    "gap_pct",
    "abs_gap_pct",
    "source_coverage_pct",
    "model_version",
    "canonical_status",
]

TIPS_INFLATION_MONTHLY_DETAIL_COLUMNS = [
    "record_date",
    "quarter",
    "event_type",
    "cohort_id",
    "cusip",
    "issue_date",
    "maturity_date",
    "issued_amt_mil",
    "inflation_adjustment_mil",
    "prior_inflation_adjustment_mil",
    "stock_delta_inflation_adjustment_mil",
    "issue_adjustment_mil",
    "redeemed_amt_mil",
    "prior_redeemed_amt_mil",
    "redeemed_delta_mil",
    "redeemed_increment_mil",
    "redemption_adjustment_mil",
    "maturity_floor_adjustment_mil",
    "modeled_inflation_compensation_mil",
    "is_first_observed_cohort_month",
    "source_coverage_status",
    "model_version",
    "canonical_status",
]

TIPS_INFLATION_RECONCILIATION_COLUMNS = [
    "quarter",
    "official_interest_mil",
    "modeled_interest_mil",
    "gap_mil",
    "abs_gap_mil",
    "gap_pct",
    "abs_gap_pct",
    "source_coverage_pct",
    "model_version",
    "canonical_status",
]

TIPS_COUPON_FLOW_DETAIL_COLUMNS = [
    "quarter",
    "component",
    "cash_coupon_mil",
    "accrued_interest_open_mil",
    "accrued_interest_close_mil",
    "issue_accrued_interest_mil",
    "modeled_interest_mil",
    "source_coverage_status",
    "model_version",
    "canonical_status",
]

NONBILL_DISCOUNT_PREMIUM_DETAIL_COLUMNS = [
    "schema_version",
    "lot_id",
    "cusip",
    "instrument_family",
    "auction_date",
    "issue_date",
    "contract_end_date",
    "is_reopening",
    "price_source_tier",
    "clean_price_per100",
    "par_mil",
    "coupon_rate_decimal",
    "reported_yield_pct",
    "solved_effective_yield_pct",
    "period_start",
    "period_end",
    "period_fraction",
    "opening_cv_per100",
    "effective_interest_per100",
    "stated_coupon_per100",
    "amortization_per100",
    "quarter",
    "overlap_days",
    "period_days",
    "quarter_amortization_mil",
    "dp_sign",
    "terminal_residual_per100",
    "source_identity_status",
    "model_version",
    "certification_status",
]

INTEREST_COMPONENT_CERTIFICATION_COLUMNS = [
    "schema_version",
    "run_id",
    "quarter",
    "calendar_year",
    "holdout_flag",
    "scope_id",
    "component_id",
    "instrument_family",
    "model_version",
    "official_interest_mil",
    "model_interest_mil",
    "gap_mil",
    "abs_gap_mil",
    "ape_pct",
    "source_coverage_pct",
    "source_identity_status",
    "source_price_tier",
    "lot_count",
    "conservation_residual_mil",
    "terminal_carrying_residual_max_per100",
    "fixed_control_residual_mil",
    "timing_status",
    "quarter_gate_status",
    "annual_gate_status",
    "holdout_gate_status",
    "certification_status",
    "included_in_scope",
    "exclusion_reason_code",
    "failure_code",
    "evidence_artifact",
    "input_sha256",
    "output_sha256",
]

INTEREST_SCOPE_CERTIFICATION_COLUMNS = [
    "schema_version",
    "run_id",
    "quarter",
    "scope_id",
    "scope_claim",
    "expected_components",
    "present_components",
    "certified_components",
    "excluded_components",
    "official_scope_total_mil",
    "model_scope_total_mil",
    "official_excluded_amount_mil",
    "gap_mil",
    "ape_pct",
    "completeness_status",
    "aggregate_gate_status",
    "fail_closed_status",
    "certification_status",
    "failure_code",
    "component_manifest_sha256",
]

SECTOR_INTEREST_ALLOCATION_COLUMNS = [
    "schema_version",
    "quarter",
    "scope_id",
    "tdc_sector",
    "tdcsim_holder",
    "component_id",
    "component_in_certified_core_scope",
    "component_in_extended_scope",
    "weight_source",
    "weight_time_basis",
    "weight_model_version",
    "aggregate_control_basis",
    "control_quality_tier",
    "component_certification_status",
    "official_interest_mil",
    "model_interest_mil",
    "raw_weight_mil",
    "component_weight_total_mil",
    "attributed_weight_mil",
    "residual_weight_mil",
    "attributed_weight_coverage_pct",
    "allocation_share",
    "allocated_official_interest_mil",
    "allocated_model_interest_mil",
    "selected_allocated_interest_mil",
    "allocation_status",
]

_PRIVATE_PREFIX = "Private"
_DEFAULT_SOURCE_STATUS = "materialized_portfolio_snapshot"
BILL_FLOW_MODEL_VERSION = "bill_auction_lot_discount_flow_v1"
FIXED_COUPON_FLOW_MODEL_VERSION = "fixed_coupon_mspd_auction_event_accrual_v1"
FRN_FLOW_MODEL_VERSION = "frn_auction_lot_benchmark_accrual_v1"
TIPS_INFLATION_FLOW_MODEL_VERSION = "tips_mspd_cusip_stock_event_flow_v3"
TIPS_COUPON_FLOW_MODEL_VERSION = "tips_coupon_accrued_liability_flow_v1"
NONBILL_DISCOUNT_PREMIUM_MODEL_VERSION = "nonbill_effective_interest_discount_premium_v1"
INTEREST_COMPONENT_CERTIFICATION_SCHEMA_VERSION = "interest_component_certification_v1"
INTEREST_SCOPE_CERTIFICATION_SCHEMA_VERSION = "interest_scope_certification_v1"
SECTOR_INTEREST_ALLOCATION_SCHEMA_VERSION = "sector_interest_allocation_v1"
CERTIFIED_QUARTERLY_SCOPE_ID = "marketable_quarterly_certified_core_ex_tips_coupon_nonbill_dp_v1"
SECTOR_INTEREST_CERTIFIED_CORE_SCOPE_ID = "certified_core_ex_tips_coupon_nonbill_dp"
SECTOR_INTEREST_EXTENDED_TIPS_COUPON_SCOPE_ID = "extended_with_timing_caveated_tips_coupon"
_SECTOR_INTEREST_CORE_COMPONENTS = {
    "bill_discount",
    "coupon_accrual",
    "frn_interest",
    "tips_inflation_compensation",
}
_SECTOR_INTEREST_EXTENDED_COMPONENTS = _SECTOR_INTEREST_CORE_COMPONENTS | {"tips_coupon_accrual"}
_UNALLOCATED_SECTOR = "Unallocated"
_US_FEDERAL_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def build_quarterly_interest_detail(
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    """Build a quarter/security/component interest detail table from snapshots."""

    frames: list[pd.DataFrame] = []
    for quarter, snapshot in _iter_snapshots(portfolio_snapshots):
        quarter_detail = _build_quarter_interest_detail_frame(quarter, snapshot)
        if not quarter_detail.empty:
            frames.append(quarter_detail)
    if not frames:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    detail = pd.concat(frames, ignore_index=True)
    detail["days_held"] = pd.to_numeric(detail["days_held"], errors="coerce").fillna(0).astype(int)
    detail["excluded_from_default_canonical"] = detail["excluded_from_default_canonical"].astype(bool)
    return detail.sort_values(
        ["quarter", "broad_holder_class", "tdcsim_holder", "cusip/cohort_id", "component"],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_stock_only_interest(
    detail: pd.DataFrame,
    *,
    holder_column: str = "tdcsim_holder",
    include_excluded: bool = False,
) -> pd.DataFrame:
    """Aggregate stock-only interest detail by quarter and holder."""

    _require_columns(detail, ["quarter", holder_column, "interest_amount"])
    working = detail.copy()
    if not include_excluded and "excluded_from_default_canonical" in working.columns:
        working = working.loc[~working["excluded_from_default_canonical"].fillna(False)].copy()
    if working.empty:
        return pd.DataFrame(columns=["quarter", holder_column, "stock_only_interest_proxy"])
    aggregated = (
        working.groupby(["quarter", holder_column], dropna=False, sort=True)["interest_amount"]
        .sum()
        .reset_index()
        .rename(columns={"interest_amount": "stock_only_interest_proxy"})
    )
    return aggregated


def build_sector_interest_allocation(
    interest_detail: pd.DataFrame,
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
    component_certification: pd.DataFrame,
    *,
    bill_interest_flow_detail: pd.DataFrame | None = None,
    frn_interest_flow_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Allocate aggregate interest controls across replay sectors by component weights."""

    weights = _build_sector_interest_allocation_weights(
        interest_detail,
        portfolio_snapshots,
        bill_interest_flow_detail=bill_interest_flow_detail,
        frn_interest_flow_detail=frn_interest_flow_detail,
    )
    if interest_detail is not None and not interest_detail.empty and "quarter" in interest_detail.columns:
        allowed_quarters = set(interest_detail["quarter"].dropna().astype(str).unique().tolist())
        weights = weights.loc[weights["quarter"].astype(str).isin(allowed_quarters)].copy()
    controls = _build_sector_interest_component_controls(component_certification, weights)
    if controls.empty or weights.empty:
        return pd.DataFrame(columns=SECTOR_INTEREST_ALLOCATION_COLUMNS)

    rows = []
    for _, control in controls.iterrows():
        quarter = str(control["quarter"])
        component_id = str(control["component_id"])
        component_weights = weights.loc[
            weights["quarter"].astype(str).eq(quarter)
            & weights["component_id"].astype(str).eq(component_id)
        ].copy()
        if component_weights.empty:
            continue
        weight_total = pd.to_numeric(component_weights["raw_weight_mil"], errors="coerce").sum()
        if pd.isna(weight_total) or abs(float(weight_total)) <= 1.0e-12:
            continue
        unallocated_mask = component_weights["tdcsim_holder"].map(_sector_interest_tdc_sector).eq(_UNALLOCATED_SECTOR)
        attributed_weight = pd.to_numeric(
            component_weights.loc[~unallocated_mask, "raw_weight_mil"],
            errors="coerce",
        ).sum()
        residual_weight = pd.to_numeric(
            component_weights.loc[unallocated_mask, "raw_weight_mil"],
            errors="coerce",
        ).sum()
        coverage_pct = float(attributed_weight) / float(weight_total) * 100.0 if float(weight_total) else np.nan
        official_interest = pd.to_numeric(control.get("official_interest_mil"), errors="coerce")
        model_interest = pd.to_numeric(control.get("model_interest_mil"), errors="coerce")
        aggregate_basis = _sector_interest_aggregate_basis(official_interest, model_interest)
        certification_status = str(control.get("certification_status", "stock_weight_proxy_no_component_certification"))
        control_quality_tier = _sector_interest_control_quality_tier(aggregate_basis, certification_status)
        for _, weight in component_weights.iterrows():
            raw_weight = float(weight["raw_weight_mil"])
            share = raw_weight / float(weight_total)
            official_alloc = official_interest * share if pd.notna(official_interest) else np.nan
            model_alloc = model_interest * share if pd.notna(model_interest) else np.nan
            selected_alloc = official_alloc
            if pd.isna(selected_alloc):
                selected_alloc = model_alloc
            if pd.isna(selected_alloc):
                selected_alloc = raw_weight
            rows.append(
                {
                    "schema_version": SECTOR_INTEREST_ALLOCATION_SCHEMA_VERSION,
                    "quarter": quarter,
                    "scope_id": _sector_interest_component_scope_id(component_id),
                    "tdc_sector": _sector_interest_tdc_sector(weight["tdcsim_holder"]),
                    "tdcsim_holder": weight["tdcsim_holder"],
                    "component_id": component_id,
                    "component_in_certified_core_scope": component_id in _SECTOR_INTEREST_CORE_COMPONENTS,
                    "component_in_extended_scope": component_id in _SECTOR_INTEREST_EXTENDED_COMPONENTS,
                    "weight_source": weight["weight_source"],
                    "weight_time_basis": weight.get("weight_time_basis", "unspecified"),
                    "weight_model_version": weight.get("weight_model_version", "unspecified"),
                    "aggregate_control_basis": aggregate_basis,
                    "control_quality_tier": control_quality_tier,
                    "component_certification_status": certification_status,
                    "official_interest_mil": official_interest,
                    "model_interest_mil": model_interest,
                    "raw_weight_mil": raw_weight,
                    "component_weight_total_mil": float(weight_total),
                    "attributed_weight_mil": float(attributed_weight),
                    "residual_weight_mil": float(residual_weight),
                    "attributed_weight_coverage_pct": coverage_pct,
                    "allocation_share": share,
                    "allocated_official_interest_mil": official_alloc,
                    "allocated_model_interest_mil": model_alloc,
                    "selected_allocated_interest_mil": selected_alloc,
                    "allocation_status": _sector_interest_allocation_status(aggregate_basis, certification_status),
                }
            )
    if not rows:
        return pd.DataFrame(columns=SECTOR_INTEREST_ALLOCATION_COLUMNS)
    return (
        pd.DataFrame(rows, columns=SECTOR_INTEREST_ALLOCATION_COLUMNS)
        .sort_values(["quarter", "component_id", "tdcsim_holder"], kind="stable")
        .reset_index(drop=True)
    )


def build_sector_interest_totals(sector_interest_allocation: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sector interest allocation rows to quarter/sector totals."""

    columns = [
        "quarter",
        "scope_id",
        "tdc_sector",
        "selected_allocated_interest_mil",
        "allocated_official_interest_mil",
        "allocated_model_interest_mil",
        "component_count",
        "tdcsim_holder_count",
        "official_control_component_count",
        "certified_component_count",
        "min_attributed_weight_coverage_pct",
        "allocation_statuses",
    ]
    if sector_interest_allocation is None or sector_interest_allocation.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        sector_interest_allocation,
        [
            "quarter",
            "scope_id",
            "tdc_sector",
            "tdcsim_holder",
            "component_id",
            "component_in_certified_core_scope",
            "component_in_extended_scope",
            "selected_allocated_interest_mil",
            "allocated_official_interest_mil",
            "allocated_model_interest_mil",
            "aggregate_control_basis",
            "component_certification_status",
            "attributed_weight_coverage_pct",
            "allocation_status",
        ],
    )
    working = sector_interest_allocation.copy()
    for column in [
        "selected_allocated_interest_mil",
        "allocated_official_interest_mil",
        "allocated_model_interest_mil",
        "attributed_weight_coverage_pct",
    ]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    scope_frames: list[pd.DataFrame] = []
    scope_specs = [
        (SECTOR_INTEREST_CERTIFIED_CORE_SCOPE_ID, "component_in_certified_core_scope"),
        (SECTOR_INTEREST_EXTENDED_TIPS_COUPON_SCOPE_ID, "component_in_extended_scope"),
    ]
    for scope_id, flag_column in scope_specs:
        scoped = working.loc[working[flag_column].fillna(False).astype(bool)].copy()
        if scoped.empty:
            continue
        grouped = (
            scoped.groupby(["quarter", "tdc_sector"], dropna=False, sort=True)
            .agg(
                selected_allocated_interest_mil=(
                    "selected_allocated_interest_mil",
                    lambda values: values.sum(min_count=1),
                ),
                allocated_official_interest_mil=(
                    "allocated_official_interest_mil",
                    lambda values: values.sum(min_count=1),
                ),
                allocated_model_interest_mil=(
                    "allocated_model_interest_mil",
                    lambda values: values.sum(min_count=1),
                ),
                component_count=("component_id", "nunique"),
                tdcsim_holder_count=("tdcsim_holder", "nunique"),
                min_attributed_weight_coverage_pct=("attributed_weight_coverage_pct", "min"),
                allocation_statuses=("allocation_status", lambda values: ";".join(sorted(set(map(str, values))))),
            )
            .reset_index()
        )
        component_flags = scoped.drop_duplicates(["quarter", "tdc_sector", "component_id"]).copy()
        count_flags = (
            component_flags.assign(
                official_control_component=component_flags["aggregate_control_basis"]
                .astype(str)
                .eq("official_treasury_component_pool")
                .astype(int),
                certified_component=component_flags["component_certification_status"]
                .astype(str)
                .eq("certified_quarterly")
                .astype(int),
            )
            .groupby(["quarter", "tdc_sector"], dropna=False, sort=True)
            .agg(
                official_control_component_count=("official_control_component", "sum"),
                certified_component_count=("certified_component", "sum"),
            )
            .reset_index()
        )
        grouped = grouped.merge(count_flags, on=["quarter", "tdc_sector"], how="left")
        grouped["scope_id"] = scope_id
        scope_frames.append(grouped.loc[:, columns])
    if not scope_frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(scope_frames, ignore_index=True, sort=False).sort_values(
        ["quarter", "scope_id", "tdc_sector"],
        kind="stable",
    ).reset_index(drop=True)


def _sector_interest_tdc_sector(holder: object) -> str:
    value = str(holder).strip()
    if value in {"SourceBasisResidual", "Unknown", _UNALLOCATED_SECTOR}:
        return _UNALLOCATED_SECTOR
    if value.startswith(f"{_PRIVATE_PREFIX}:"):
        return _PRIVATE_PREFIX
    return value or "Unknown"


def _build_sector_interest_allocation_weights(
    interest_detail: pd.DataFrame,
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    bill_interest_flow_detail: pd.DataFrame | None = None,
    frn_interest_flow_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    use_bill_flow = bill_interest_flow_detail is not None and not bill_interest_flow_detail.empty
    use_frn_flow = frn_interest_flow_detail is not None and not frn_interest_flow_detail.empty
    frames = [
        _sector_interest_weights_from_detail(
            interest_detail,
            include_bill=not use_bill_flow,
        ),
        _sector_interest_bill_flow_weights(bill_interest_flow_detail, portfolio_snapshots) if use_bill_flow else pd.DataFrame(),
        _sector_interest_frn_flow_weights(frn_interest_flow_detail, portfolio_snapshots) if use_frn_flow else pd.DataFrame(),
        _sector_interest_weights_from_portfolio(
            portfolio_snapshots,
            include_frn=not use_frn_flow,
        ),
    ]
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    weight_columns = [
        "quarter",
        "tdcsim_holder",
        "component_id",
        "raw_weight_mil",
        "weight_source",
        "weight_time_basis",
        "weight_model_version",
    ]
    if not frames:
        return pd.DataFrame(columns=weight_columns)
    weights = pd.concat(frames, ignore_index=True, sort=False)
    weights["raw_weight_mil"] = pd.to_numeric(weights["raw_weight_mil"], errors="coerce").fillna(0.0)
    weights = weights.loc[weights["raw_weight_mil"].abs() > 1.0e-12].copy()
    weights["weight_time_basis"] = weights["weight_time_basis"].fillna("unspecified")
    weights["weight_model_version"] = weights["weight_model_version"].fillna("unspecified")
    return (
        weights.groupby(
            [
                "quarter",
                "tdcsim_holder",
                "component_id",
                "weight_source",
                "weight_time_basis",
                "weight_model_version",
            ],
            dropna=False,
            sort=True,
        )[
            "raw_weight_mil"
        ]
        .sum()
        .reset_index()
        .loc[:, weight_columns]
    )


def _sector_interest_weights_from_detail(
    interest_detail: pd.DataFrame,
    *,
    include_bill: bool = True,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "tdcsim_holder",
        "component_id",
        "raw_weight_mil",
        "weight_source",
        "weight_time_basis",
        "weight_model_version",
    ]
    if interest_detail is None or interest_detail.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(interest_detail, ["quarter", "tdcsim_holder", "component", "interest_amount"])
    component_map = {
        "fixed_coupon_accrual": (
            "coupon_accrual",
            "stock_interest_detail_fixed_coupon_weight",
            "quarter_exposure_detail",
            "stock_interest_detail_v1",
        ),
        "tips_coupon_accrual": (
            "tips_coupon_accrual",
            "stock_interest_detail_tips_coupon_weight",
            "quarter_exposure_detail",
            "stock_interest_detail_v1",
        ),
    }
    if include_bill:
        component_map["bill_discount_amortization"] = (
            "bill_discount",
            "stock_interest_detail_bill_discount_weight",
            "quarter_end_survivor_stock_detail",
            "stock_interest_detail_v1",
        )
    working = interest_detail.copy()
    if "excluded_from_default_canonical" in working.columns:
        excluded = working["excluded_from_default_canonical"].fillna(False).astype(bool)
        unallocated_evidence = working["tdcsim_holder"].map(_sector_interest_tdc_sector).eq(_UNALLOCATED_SECTOR)
        working = working.loc[~excluded | unallocated_evidence].copy()
    working["component_id"] = working["component"].map(lambda value: component_map.get(str(value), (pd.NA, pd.NA, pd.NA, pd.NA))[0])
    working["weight_source"] = working["component"].map(lambda value: component_map.get(str(value), (pd.NA, pd.NA, pd.NA, pd.NA))[1])
    working["weight_time_basis"] = working["component"].map(lambda value: component_map.get(str(value), (pd.NA, pd.NA, pd.NA, pd.NA))[2])
    working["weight_model_version"] = working["component"].map(lambda value: component_map.get(str(value), (pd.NA, pd.NA, pd.NA, pd.NA))[3])
    working = working.loc[working["component_id"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["raw_weight_mil"] = pd.to_numeric(working["interest_amount"], errors="coerce").fillna(0.0).abs()
    return (
        working.groupby(
            ["quarter", "tdcsim_holder", "component_id", "weight_source", "weight_time_basis", "weight_model_version"],
            dropna=False,
            sort=True,
        )[
            "raw_weight_mil"
        ]
        .sum()
        .reset_index()
        .loc[:, columns]
    )


def _sector_interest_weights_from_portfolio(
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    include_frn: bool = True,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "tdcsim_holder",
        "component_id",
        "raw_weight_mil",
        "weight_source",
        "weight_time_basis",
        "weight_model_version",
    ]
    frames: list[pd.DataFrame] = []
    for quarter, snapshot in _iter_snapshots(portfolio_snapshots):
        if snapshot is None or snapshot.empty:
            continue
        frame = snapshot.copy()
        security_type = _text_from_columns(frame, "SecurityType", "security_type").str.upper()
        holder = _sector_interest_holder_series(frame)
        face_value = _numeric_from_columns(frame, "FaceValue", "face_value", default=0.0).clip(lower=0.0)
        adjusted_principal = _numeric_from_columns(
            frame,
            "AdjustedPrincipal",
            "adjusted_principal",
            default=np.nan,
        )
        index_ratio = _numeric_from_columns(frame, "IndexRatio", "index_ratio", default=1.0).replace(0.0, np.nan)
        tips_exposure = adjusted_principal.where(adjusted_principal.notna() & (adjusted_principal > 0.0))
        tips_exposure = tips_exposure.fillna(face_value * index_ratio.fillna(1.0)).clip(lower=0.0)
        component_rows = []
        if include_frn:
            component_rows.append(
                (
                    security_type.eq("FRN"),
                    face_value,
                    "frn_interest",
                    "portfolio_frn_par_weight",
                    "quarter_end_portfolio_stock",
                    "portfolio_frn_par_weight_v1",
                )
            )
        component_rows.append(
            (
                security_type.eq("TIPS"),
                tips_exposure,
                "tips_inflation_compensation",
                "portfolio_tips_adjusted_principal_weight",
                "quarter_end_portfolio_stock",
                "portfolio_tips_adjusted_principal_weight_v1",
            )
        )
        for mask, raw_weight, component_id, weight_source, time_basis, model_version in component_rows:
            selected = pd.DataFrame(
                {
                    "quarter": str(quarter),
                    "tdcsim_holder": holder.loc[mask].to_numpy(),
                    "component_id": component_id,
                    "raw_weight_mil": raw_weight.loc[mask].to_numpy(),
                    "weight_source": weight_source,
                    "weight_time_basis": time_basis,
                    "weight_model_version": model_version,
                }
            )
            selected = selected.loc[selected["raw_weight_mil"] > 0.0]
            if not selected.empty:
                frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True, sort=False).loc[:, columns]


def _sector_interest_bill_flow_weights(
    bill_interest_flow_detail: pd.DataFrame | None,
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "tdcsim_holder",
        "component_id",
        "raw_weight_mil",
        "weight_source",
        "weight_time_basis",
        "weight_model_version",
    ]
    if bill_interest_flow_detail is None or bill_interest_flow_detail.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        bill_interest_flow_detail,
        ["quarter", "cusip", "issue_date", "maturity_date", "modeled_interest_mil"],
    )
    flow = bill_interest_flow_detail.copy()
    flow["component_id"] = "bill_discount"
    flow["flow_weight_mil"] = pd.to_numeric(flow["modeled_interest_mil"], errors="coerce").abs()
    return _sector_interest_flow_weights_from_holder_shares(
        flow,
        portfolio_snapshots,
        component_id="bill_discount",
        security_type="Fixed",
        maturity_category="bills",
        weight_source="bill_discount_flow_holder_share_weight",
        weight_time_basis="full_quarter_bill_flow_current_or_prior_snapshot_holder_share",
        weight_model_version=BILL_FLOW_MODEL_VERSION,
    ).loc[:, columns]


def _sector_interest_frn_flow_weights(
    frn_interest_flow_detail: pd.DataFrame | None,
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "tdcsim_holder",
        "component_id",
        "raw_weight_mil",
        "weight_source",
        "weight_time_basis",
        "weight_model_version",
    ]
    if frn_interest_flow_detail is None or frn_interest_flow_detail.empty:
        return pd.DataFrame(columns=columns)
    _require_columns(
        frn_interest_flow_detail,
        ["quarter", "cusip", "issue_date", "maturity_date", "modeled_interest_mil"],
    )
    flow = (
        frn_interest_flow_detail.copy()
        .assign(flow_weight_mil=lambda frame: pd.to_numeric(frame["modeled_interest_mil"], errors="coerce").abs())
        .groupby(["quarter", "cusip", "issue_date", "maturity_date"], dropna=False, sort=False)["flow_weight_mil"]
        .sum()
        .reset_index()
    )
    flow["component_id"] = "frn_interest"
    return _sector_interest_flow_weights_from_holder_shares(
        flow,
        portfolio_snapshots,
        component_id="frn_interest",
        security_type="FRN",
        maturity_category=None,
        weight_source="frn_daily_interest_flow_holder_share_weight",
        weight_time_basis="daily_frn_interest_flow_current_or_prior_snapshot_holder_share",
        weight_model_version=FRN_FLOW_MODEL_VERSION,
    ).loc[:, columns]


def _sector_interest_flow_weights_from_holder_shares(
    flow: pd.DataFrame,
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    component_id: str,
    security_type: str,
    maturity_category: str | None,
    weight_source: str,
    weight_time_basis: str,
    weight_model_version: str,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "tdcsim_holder",
        "component_id",
        "raw_weight_mil",
        "weight_source",
        "weight_time_basis",
        "weight_model_version",
    ]
    shares = _sector_interest_portfolio_holder_shares(
        portfolio_snapshots,
        security_type=security_type,
        maturity_category=maturity_category,
    )
    working = flow.copy()
    working["quarter"] = working["quarter"].astype(str)
    working["issue_date"] = pd.to_datetime(working["issue_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["maturity_date"] = pd.to_datetime(working["maturity_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["cohort_key"] = _sector_interest_cohort_key(working)
    working["flow_row_id"] = np.arange(len(working))
    working["flow_weight_mil"] = pd.to_numeric(working["flow_weight_mil"], errors="coerce").fillna(0.0).abs()
    working = working.loc[working["flow_weight_mil"] > 0.0].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    current = working.merge(
        shares,
        on=["quarter", "cohort_key"],
        how="left",
    )
    current["holder_share_source"] = "current_quarter_snapshot"
    current["source_rank"] = 0
    prior_base = working.copy()
    prior_base["quarter"] = prior_base["quarter"].map(_previous_quarter_label)
    prior = prior_base.merge(
        shares,
        on=["quarter", "cohort_key"],
        how="left",
    )
    prior["quarter"] = working.set_index("flow_row_id").loc[prior["flow_row_id"], "quarter"].to_numpy()
    prior["holder_share_source"] = "prior_quarter_snapshot"
    prior["source_rank"] = 1
    candidates = pd.concat([current, prior], ignore_index=True, sort=False)
    matched = candidates.loc[candidates["tdcsim_holder"].notna()].copy()
    if not matched.empty:
        matched = matched.loc[
            matched["source_rank"].eq(matched.groupby("flow_row_id")["source_rank"].transform("min"))
        ].copy()
    matched_ids = set(matched["flow_row_id"].dropna().astype(int).tolist()) if not matched.empty else set()
    missing = working.loc[~working["flow_row_id"].isin(matched_ids)].copy()
    if not missing.empty:
        missing["tdcsim_holder"] = _UNALLOCATED_SECTOR
        missing["holder_share"] = 1.0
        missing["holder_share_source"] = "unallocated_no_current_or_prior_snapshot_holder_share"
        matched = pd.concat([matched, missing], ignore_index=True, sort=False)
    matched["raw_weight_mil"] = pd.to_numeric(matched["flow_weight_mil"], errors="coerce").fillna(0.0) * pd.to_numeric(
        matched["holder_share"],
        errors="coerce",
    ).fillna(0.0)
    matched["component_id"] = component_id
    matched["weight_source"] = weight_source
    matched["weight_time_basis"] = matched["holder_share_source"].map(
        lambda source: f"{weight_time_basis}:{source}"
    )
    matched["weight_model_version"] = weight_model_version
    return (
        matched.groupby(
            ["quarter", "tdcsim_holder", "component_id", "weight_source", "weight_time_basis", "weight_model_version"],
            dropna=False,
            sort=True,
        )["raw_weight_mil"]
        .sum()
        .reset_index()
        .loc[:, columns]
    )


def _sector_interest_portfolio_holder_shares(
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
    *,
    security_type: str,
    maturity_category: str | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for quarter, snapshot in _iter_snapshots(portfolio_snapshots):
        if snapshot is None or snapshot.empty:
            continue
        frame = snapshot.copy()
        frame_security_type = _text_from_columns(frame, "SecurityType", "security_type").str.upper()
        mask = frame_security_type.eq(str(security_type).upper())
        if maturity_category is not None:
            category = _text_from_columns(frame, "MaturityCategory", "maturity_category").str.lower()
            mask &= category.eq(str(maturity_category).lower())
        if not mask.any():
            continue
        selected = frame.loc[mask].copy()
        selected["quarter"] = str(quarter)
        selected["issue_date"] = pd.to_datetime(
            _value_from_columns(selected, "IssueDate", "issue_date"),
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        selected["maturity_date"] = pd.to_datetime(
            _value_from_columns(selected, "MaturityDate", "maturity_date"),
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
        selected["cohort_key"] = _sector_interest_cohort_key(selected)
        selected["tdcsim_holder"] = _sector_interest_holder_series(selected)
        selected["holder_weight_mil"] = _numeric_from_columns(selected, "FaceValue", "face_value", default=0.0).clip(lower=0.0)
        selected = selected.loc[selected["holder_weight_mil"] > 0.0].copy()
        if not selected.empty:
            frames.append(selected[["quarter", "cohort_key", "tdcsim_holder", "holder_weight_mil"]])
    if not frames:
        return pd.DataFrame(columns=["quarter", "cohort_key", "tdcsim_holder", "holder_share"])
    shares = (
        pd.concat(frames, ignore_index=True, sort=False)
        .groupby(["quarter", "cohort_key", "tdcsim_holder"], dropna=False, sort=True)["holder_weight_mil"]
        .sum()
        .reset_index()
    )
    totals = shares.groupby(["quarter", "cohort_key"], dropna=False, sort=True)["holder_weight_mil"].transform("sum")
    shares["holder_share"] = shares["holder_weight_mil"] / totals.replace(0.0, np.nan)
    return shares.loc[shares["holder_share"].notna(), ["quarter", "cohort_key", "tdcsim_holder", "holder_share"]]


def _sector_interest_cohort_key(frame: pd.DataFrame) -> pd.Series:
    return (
        _text_from_columns(frame, "cusip")
        + "|"
        + _text_from_columns(frame, "issue_date", "IssueDate")
        + "|"
        + _text_from_columns(frame, "maturity_date", "MaturityDate")
    )


def _previous_quarter_label(quarter: object) -> str:
    try:
        return str(pd.Period(str(quarter), freq="Q") - 1)
    except Exception:
        return str(quarter)


def _sector_interest_holder_series(frame: pd.DataFrame) -> pd.Series:
    tdcsim_holder = _text_from_columns(frame, "tdcsim_holder", "HolderType", "holder_type")
    broad_holder = _text_from_columns(frame, "broad_holder_class")
    holder_subbucket = _text_from_columns(frame, "tdcsim_holder_subbucket", "HolderSubBucket", "holder_subbucket")
    holder = tdcsim_holder.where(tdcsim_holder.ne(""), broad_holder)
    private_with_subbucket = holder.eq(_PRIVATE_PREFIX) & holder_subbucket.ne("")
    holder = holder.where(~private_with_subbucket, _PRIVATE_PREFIX + ":" + holder_subbucket)
    return holder.replace("", pd.NA).fillna("Unknown")


def _build_sector_interest_component_controls(
    component_certification: pd.DataFrame,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "component_id",
        "official_interest_mil",
        "model_interest_mil",
        "certification_status",
    ]
    if weights is None or weights.empty:
        return pd.DataFrame(columns=columns)
    control_components = sorted(weights["component_id"].dropna().astype(str).unique().tolist())
    if component_certification is None or component_certification.empty:
        controls = pd.DataFrame(columns=columns)
    else:
        _require_columns(component_certification, ["quarter", "component_id", "official_interest_mil", "model_interest_mil"])
        controls = component_certification.copy()
        if "certification_status" not in controls.columns:
            controls["certification_status"] = "component_control_present_status_unspecified"
        controls = controls.loc[controls["component_id"].astype(str).isin(control_components), columns].copy()
    weight_controls = (
        weights.groupby(["quarter", "component_id"], dropna=False, sort=True)
        .agg(
            raw_weight_mil=("raw_weight_mil", "sum"),
            has_interest_unit_weight=(
                "weight_source",
                lambda values: any(
                    token in str(value)
                    for value in values
                    for token in ["interest_detail", "flow_holder_share_weight"]
                ),
            ),
        )
        .reset_index()
    )
    merged = weight_controls.merge(controls, on=["quarter", "component_id"], how="left")
    merged["model_interest_mil"] = pd.to_numeric(merged["model_interest_mil"], errors="coerce")
    merged["official_interest_mil"] = pd.to_numeric(merged["official_interest_mil"], errors="coerce")
    missing_control = (
        merged["official_interest_mil"].isna()
        & merged["model_interest_mil"].isna()
        & merged["has_interest_unit_weight"].fillna(False).astype(bool)
    )
    merged.loc[missing_control, "model_interest_mil"] = merged.loc[missing_control, "raw_weight_mil"]
    merged["certification_status"] = merged["certification_status"].fillna("stock_weight_proxy_no_component_certification")
    return merged.loc[:, columns]


def _sector_interest_aggregate_basis(official_interest, model_interest) -> str:
    if pd.notna(official_interest):
        return "official_treasury_component_pool"
    if pd.notna(model_interest):
        return "tdcsim_model_or_stock_component_pool"
    return "stock_weight_proxy_component_pool"


def _sector_interest_allocation_status(aggregate_basis: str, certification_status: str) -> str:
    if aggregate_basis == "official_treasury_component_pool" and certification_status == "certified_quarterly":
        return "official_control_certified_component"
    if aggregate_basis == "official_treasury_component_pool":
        return "official_control_uncertified_component"
    return "tdcsim_stock_model_proxy_component"


def _sector_interest_control_quality_tier(aggregate_basis: str, certification_status: str) -> str:
    if aggregate_basis == "official_treasury_component_pool" and certification_status == "certified_quarterly":
        return "certified_official"
    if aggregate_basis == "official_treasury_component_pool":
        return "uncertified_official"
    if aggregate_basis == "tdcsim_model_or_stock_component_pool":
        return "tdcsim_model_proxy"
    return "no_aggregate_control"


def _sector_interest_component_scope_id(component_id: str) -> str:
    if component_id in _SECTOR_INTEREST_CORE_COMPONENTS:
        return SECTOR_INTEREST_CERTIFIED_CORE_SCOPE_ID
    if component_id in _SECTOR_INTEREST_EXTENDED_COMPONENTS:
        return SECTOR_INTEREST_EXTENDED_TIPS_COUPON_SCOPE_ID
    return "component_not_in_sector_interest_scope"


def align_stock_only_interest_proxy(
    detail: pd.DataFrame,
    *,
    reference: pd.DataFrame | None = None,
    holder_column: str = "tdcsim_holder",
    reference_mapping: Mapping[str, str] | None = None,
    reference_quarter_column: str = "quarter",
    include_excluded: bool = False,
) -> pd.DataFrame:
    """Aggregate stock-only interest and optionally align it to reference columns."""

    proxy = aggregate_stock_only_interest(
        detail,
        holder_column=holder_column,
        include_excluded=include_excluded,
    )
    if reference is None:
        return proxy
    if not reference_mapping:
        raise ValueError("reference_mapping is required when a reference frame is supplied")
    _require_columns(reference, [reference_quarter_column, *reference_mapping.values()])

    aligned_frames: list[pd.DataFrame] = []
    for holder_value, reference_column in reference_mapping.items():
        holder_proxy = proxy.loc[proxy[holder_column].astype(str) == str(holder_value)].copy()
        holder_proxy = holder_proxy.rename(columns={"stock_only_interest_proxy": "stock_only_interest_proxy"})
        holder_reference = reference[[reference_quarter_column, reference_column]].copy()
        holder_reference = holder_reference.rename(
            columns={
                reference_quarter_column: "quarter",
                reference_column: "reference_interest_proxy",
            }
        )
        merged = holder_reference.merge(holder_proxy, on="quarter", how="outer")
        merged[holder_column] = holder_value
        merged["reference_column"] = reference_column
        merged["difference_vs_reference"] = (
            pd.to_numeric(merged["stock_only_interest_proxy"], errors="coerce").fillna(0.0)
            - pd.to_numeric(merged["reference_interest_proxy"], errors="coerce").fillna(0.0)
        )
        aligned_frames.append(merged)
    if not aligned_frames:
        return pd.DataFrame(
            columns=[
                "quarter",
                holder_column,
                "reference_column",
                "reference_interest_proxy",
                "stock_only_interest_proxy",
                "difference_vs_reference",
            ]
        )
    return (
        pd.concat(aligned_frames, ignore_index=True)
        .sort_values(["quarter", holder_column], kind="stable")
        .reset_index(drop=True)
    )


def build_treasury_interest_expense_diagnostic(
    treasury_interest_expense: pd.DataFrame,
    interest_detail: pd.DataFrame,
    *,
    tier2_interest_candidate: pd.DataFrame | None = None,
    bill_interest_flow_detail: pd.DataFrame | None = None,
    fixed_coupon_flow_detail: pd.DataFrame | None = None,
    frn_interest_flow_detail: pd.DataFrame | None = None,
    tips_inflation_flow_detail: pd.DataFrame | None = None,
    tips_coupon_flow_detail: pd.DataFrame | None = None,
    nonbill_discount_premium_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare official Treasury interest pools to replay and TDC-EST diagnostics."""

    official = _official_treasury_interest_pools(treasury_interest_expense)
    replay = _replay_interest_pools(interest_detail)
    flow = pd.concat(
        [
            _bill_flow_interest_pool(bill_interest_flow_detail),
            _fixed_coupon_flow_interest_pool(fixed_coupon_flow_detail),
            _frn_flow_interest_pool(frn_interest_flow_detail),
            _tips_inflation_flow_interest_pool(tips_inflation_flow_detail),
            _tips_coupon_flow_interest_pool(tips_coupon_flow_detail),
            _nonbill_discount_premium_flow_interest_pool(nonbill_discount_premium_detail),
        ],
        ignore_index=True,
    )
    if not flow.empty:
        replay = replay.loc[~replay["official_pool"].isin(flow["official_pool"].dropna().unique())].copy()
        replay = pd.concat([replay, flow], ignore_index=True)
    candidate = _tdcest_candidate_interest_pools(tier2_interest_candidate)
    merged = official.merge(replay, on=["quarter", "official_pool"], how="outer")
    merged = merged.merge(candidate, on=["quarter", "official_pool"], how="outer")
    for column in [
        "official_interest_expense_mil",
        "official_row_count",
        "replay_component_interest_mil",
        "tdcest_candidate_component_mil",
    ]:
        if column not in merged.columns:
            merged[column] = np.nan
    merged["replay_minus_official_mil"] = (
        pd.to_numeric(merged["replay_component_interest_mil"], errors="coerce")
        - pd.to_numeric(merged["official_interest_expense_mil"], errors="coerce")
    )
    merged["tdcest_candidate_minus_official_mil"] = (
        pd.to_numeric(merged["tdcest_candidate_component_mil"], errors="coerce")
        - pd.to_numeric(merged["official_interest_expense_mil"], errors="coerce")
    )
    merged["diagnostic_status"] = np.where(
        merged["official_interest_expense_mil"].notna(),
        "official_pool_present_diagnostic_only",
        "no_official_pool_for_component",
    )
    if merged.empty:
        return pd.DataFrame(columns=TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS)
    return (
        merged.loc[:, TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS]
        .sort_values(["quarter", "official_pool"], kind="stable")
        .reset_index(drop=True)
    )


def build_bill_interest_flow_detail(
    auction_lots: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build aggregate bill-discount interest flows from auction lots."""

    if auction_lots is None or auction_lots.empty:
        return pd.DataFrame(columns=BILL_INTEREST_FLOW_DETAIL_COLUMNS)
    required = {"lot_id", "cusip", "issue_date", "maturity_date", "price_per100", "total_accepted"}
    _require_columns(auction_lots, sorted(required))
    rows: list[dict[str, object]] = []
    window_start = _quarter_start(start_quarter) if start_quarter else None
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else None
    for _, row in auction_lots.iterrows():
        issue_date = pd.to_datetime(row.get("issue_date"), errors="coerce")
        maturity_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        price_per100 = _num(row.get("price_per100"), default=np.nan)
        total_accepted = _num(row.get("total_accepted"), default=np.nan)
        if pd.isna(issue_date) or pd.isna(maturity_date) or pd.isna(price_per100) or pd.isna(total_accepted):
            continue
        issue_date = issue_date.normalize()
        maturity_date = maturity_date.normalize()
        term_days = int((maturity_date - issue_date).days)
        if term_days <= 0:
            continue
        lot_start = issue_date
        lot_end = maturity_date
        if window_start is not None and lot_end <= window_start:
            continue
        if window_end is not None and lot_start >= window_end:
            continue

        total_accepted_mil = total_accepted / 1_000_000.0
        price_ratio = price_per100 / 100.0
        issue_proceeds_mil = total_accepted_mil * price_ratio
        lifetime_discount_mil = total_accepted_mil - issue_proceeds_mil
        offering_amt_mil = _optional_mil(row.get("offering_amt"))
        accepted_component_sum_mil = _optional_mil(row.get("accepted_component_sum"))
        accepted_component_residual_mil = (
            total_accepted_mil - accepted_component_sum_mil
            if pd.notna(accepted_component_sum_mil)
            else np.nan
        )
        for period in _overlapping_quarters(lot_start, lot_end, start_quarter, end_quarter):
            quarter_start = period.start_time.normalize()
            quarter_end_exclusive = (period + 1).start_time.normalize()
            accrual_start = max(lot_start, quarter_start)
            accrual_end = min(lot_end, quarter_end_exclusive)
            overlap_days = int((accrual_end - accrual_start).days)
            if overlap_days <= 0:
                continue
            rows.append(
                {
                    "quarter": str(period),
                    "component": "bill_discount",
                    "lot_id": row.get("lot_id"),
                    "cusip": row.get("cusip"),
                    "auction_date": row.get("auction_date"),
                    "issue_date": issue_date.strftime("%Y-%m-%d"),
                    "maturity_date": maturity_date.strftime("%Y-%m-%d"),
                    "security_term": row.get("security_term"),
                    "par_basis": "total_accepted",
                    "total_accepted_mil": total_accepted_mil,
                    "offering_amt_mil": offering_amt_mil,
                    "accepted_component_sum_mil": accepted_component_sum_mil,
                    "accepted_component_residual_mil": accepted_component_residual_mil,
                    "price_per100": price_per100,
                    "issue_proceeds_mil": issue_proceeds_mil,
                    "lifetime_discount_mil": lifetime_discount_mil,
                    "accrual_start": accrual_start.strftime("%Y-%m-%d"),
                    "accrual_end_exclusive": accrual_end.strftime("%Y-%m-%d"),
                    "term_days": term_days,
                    "overlap_days": overlap_days,
                    "modeled_interest_mil": lifetime_discount_mil * overlap_days / float(term_days),
                    "source_coverage_status": "auction_total_accepted_price_observed",
                    "model_version": BILL_FLOW_MODEL_VERSION,
                    "canonical_status": "canonical_aggregate_bill_discount",
                }
            )
    if not rows:
        return pd.DataFrame(columns=BILL_INTEREST_FLOW_DETAIL_COLUMNS)
    return (
        pd.DataFrame(rows, columns=BILL_INTEREST_FLOW_DETAIL_COLUMNS)
        .sort_values(["quarter", "lot_id", "accrual_start"], kind="stable")
        .reset_index(drop=True)
    )


def build_bill_lot_conservation(
    bill_interest_flow_detail: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
    tolerance_mil: float = 1.0e-6,
) -> pd.DataFrame:
    """Check that allocated bill interest conserves each lot's lifetime discount."""

    if bill_interest_flow_detail is None or bill_interest_flow_detail.empty:
        return pd.DataFrame(columns=BILL_LOT_CONSERVATION_COLUMNS)
    _require_columns(
        bill_interest_flow_detail,
        ["lot_id", "issue_date", "maturity_date", "term_days", "lifetime_discount_mil", "modeled_interest_mil"],
    )
    window_start = _quarter_start(start_quarter) if start_quarter else None
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else None
    grouped = (
        bill_interest_flow_detail.groupby("lot_id", dropna=False, sort=False)
        .agg(
            issue_date=("issue_date", "first"),
            maturity_date=("maturity_date", "first"),
            term_days=("term_days", "first"),
            lifetime_discount_mil=("lifetime_discount_mil", "first"),
            allocated_interest_mil=("modeled_interest_mil", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
        )
        .reset_index()
    )
    issue = pd.to_datetime(grouped["issue_date"], errors="coerce")
    maturity = pd.to_datetime(grouped["maturity_date"], errors="coerce")
    term_days = pd.to_numeric(grouped["term_days"], errors="coerce")
    lifetime = pd.to_numeric(grouped["lifetime_discount_mil"], errors="coerce")
    if window_start is not None:
        pre_days = (pd.Series(window_start, index=grouped.index) - issue).dt.days.clip(lower=0)
        pre_days = pre_days.where(window_start < maturity, term_days)
        pre_days = pre_days.clip(upper=term_days)
    else:
        pre_days = pd.Series(0.0, index=grouped.index)
    if window_end is not None:
        post_days = (maturity - pd.Series(window_end, index=grouped.index)).dt.days.clip(lower=0)
        post_days = post_days.where(window_end > issue, term_days)
        post_days = post_days.clip(upper=term_days)
    else:
        post_days = pd.Series(0.0, index=grouped.index)
    grouped["pre_window_unallocated_mil"] = lifetime * pre_days / term_days
    grouped["post_window_unallocated_mil"] = lifetime * post_days / term_days
    grouped["conservation_residual_mil"] = (
        grouped["lifetime_discount_mil"]
        - grouped["allocated_interest_mil"]
        - grouped["pre_window_unallocated_mil"]
        - grouped["post_window_unallocated_mil"]
    )
    grouped["conservation_pass"] = grouped["conservation_residual_mil"].abs() <= float(tolerance_mil)
    return grouped.loc[:, BILL_LOT_CONSERVATION_COLUMNS]


def build_nonbill_discount_premium_flow_detail(
    auction_lots: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build nominal Note/Bond and TIPS discount/premium amortization from source auction lots."""

    if auction_lots is None or auction_lots.empty:
        return pd.DataFrame(columns=NONBILL_DISCOUNT_PREMIUM_DETAIL_COLUMNS)
    _require_columns(
        auction_lots,
        [
            "lot_id",
            "cusip",
            "instrument_family",
            "auction_date",
            "issue_date",
            "first_interest_payment_date",
            "maturity_date",
            "called_date",
            "par_mil",
            "coupon_rate_decimal",
            "clean_price_per100",
            "price_source_tier",
        ],
    )
    rows: list[dict[str, object]] = []
    supported_families = {
        "Treasury Notes",
        "Treasury Bonds",
        "Treasury Inflation Protected Securities (TIPS)",
    }
    for _, lot in auction_lots.iterrows():
        instrument_family = str(lot.get("instrument_family", "")).strip()
        if instrument_family not in supported_families:
            continue
        issue_date = pd.to_datetime(lot.get("issue_date"), errors="coerce")
        first_coupon_date = pd.to_datetime(lot.get("first_interest_payment_date"), errors="coerce")
        maturity_date = pd.to_datetime(lot.get("maturity_date"), errors="coerce")
        called_date = pd.to_datetime(lot.get("called_date"), errors="coerce")
        contract_end = called_date if pd.notna(called_date) else maturity_date
        price = _num(lot.get("clean_price_per100"), default=np.nan)
        par_mil = _num(lot.get("par_mil"), default=np.nan)
        coupon_rate = _num(lot.get("coupon_rate_decimal"), default=np.nan)
        if (
            pd.isna(issue_date)
            or pd.isna(first_coupon_date)
            or pd.isna(contract_end)
            or pd.isna(price)
            or pd.isna(par_mil)
            or pd.isna(coupon_rate)
            or par_mil <= 0.0
        ):
            continue
        issue_date = issue_date.normalize()
        first_coupon_date = first_coupon_date.normalize()
        contract_end = contract_end.normalize()
        if contract_end <= issue_date or price <= 0.0:
            continue
        payment_dates = _semiannual_coupon_dates(issue_date, first_coupon_date, contract_end)
        if not payment_dates:
            continue
        period_fractions = _semiannual_coupon_fractions(issue_date, payment_dates)
        solved_yield = _solve_effective_annual_yield(price, coupon_rate, period_fractions)
        if pd.isna(solved_yield):
            continue
        cv = price
        period_start = issue_date
        period_rows: list[dict[str, object]] = []
        total_amortization = 0.0
        for payment_date, period_fraction in zip(payment_dates, period_fractions, strict=False):
            payment_date = pd.Timestamp(payment_date).normalize()
            period_days = int((payment_date - period_start).days)
            if period_days <= 0:
                period_rows = []
                break
            opening_cv = cv
            effective_interest = opening_cv * (solved_yield / 2.0) * period_fraction
            stated_coupon = 100.0 * (coupon_rate / 2.0) * period_fraction
            amortization = effective_interest - stated_coupon
            cv = opening_cv + amortization
            total_amortization += amortization
            for period in _overlapping_quarters(period_start, payment_date, start_quarter, end_quarter):
                quarter_start = period.start_time.normalize()
                quarter_end_exclusive = (period + 1).start_time.normalize()
                overlap_start = max(period_start, quarter_start)
                overlap_end = min(payment_date, quarter_end_exclusive)
                overlap_days = int((overlap_end - overlap_start).days)
                if overlap_days <= 0:
                    continue
                period_rows.append(
                    {
                        "schema_version": "nonbill_discount_premium_detail_v1",
                        "lot_id": lot.get("lot_id"),
                        "cusip": lot.get("cusip"),
                        "instrument_family": instrument_family,
                        "auction_date": lot.get("auction_date"),
                        "issue_date": issue_date.strftime("%Y-%m-%d"),
                        "contract_end_date": contract_end.strftime("%Y-%m-%d"),
                        "is_reopening": bool(lot.get("is_reopening", False)),
                        "price_source_tier": lot.get("price_source_tier"),
                        "clean_price_per100": price,
                        "par_mil": par_mil,
                        "coupon_rate_decimal": coupon_rate,
                        "reported_yield_pct": _num(lot.get("reported_yield_pct"), default=np.nan),
                        "solved_effective_yield_pct": solved_yield * 100.0,
                        "period_start": period_start.strftime("%Y-%m-%d"),
                        "period_end": payment_date.strftime("%Y-%m-%d"),
                        "period_fraction": period_fraction,
                        "opening_cv_per100": opening_cv,
                        "effective_interest_per100": effective_interest,
                        "stated_coupon_per100": stated_coupon,
                        "amortization_per100": amortization,
                        "quarter": str(period),
                        "overlap_days": overlap_days,
                        "period_days": period_days,
                        "quarter_amortization_mil": par_mil * amortization / 100.0 * overlap_days / period_days,
                        "dp_sign": "discount" if price < 100.0 else "premium",
                        "terminal_residual_per100": np.nan,
                        "source_identity_status": (
                            "tips_real_unit_effective_interest_source_ledger"
                            if instrument_family == "Treasury Inflation Protected Securities (TIPS)"
                            else "nominal_effective_interest_source_ledger"
                        ),
                        "model_version": NONBILL_DISCOUNT_PREMIUM_MODEL_VERSION,
                        "certification_status": "candidate_source_ledger_nonbill_discount_premium",
                    }
                )
            period_start = payment_date
        if not period_rows:
            continue
        lifetime_residual = total_amortization - (100.0 - price)
        terminal_residual = cv - 100.0
        residual = max(abs(lifetime_residual), abs(terminal_residual))
        for detail_row in period_rows:
            detail_row["terminal_residual_per100"] = residual
        rows.extend(period_rows)
    if not rows:
        return pd.DataFrame(columns=NONBILL_DISCOUNT_PREMIUM_DETAIL_COLUMNS)
    return (
        pd.DataFrame(rows, columns=NONBILL_DISCOUNT_PREMIUM_DETAIL_COLUMNS)
        .sort_values(["quarter", "instrument_family", "lot_id", "period_start"], kind="stable")
        .reset_index(drop=True)
    )


def build_fixed_coupon_interest_flow_detail(
    monthly_stocks: pd.DataFrame,
    auction_events: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build source-only monthly nominal Notes/Bonds coupon accrual."""

    if monthly_stocks is None or monthly_stocks.empty:
        return pd.DataFrame(columns=FIXED_COUPON_MONTHLY_DETAIL_COLUMNS)
    _require_columns(
        monthly_stocks,
        [
            "record_date",
            "cusip",
            "security_class",
            "outstanding_mil",
            "coupon_rate_decimal",
            "maturity_date",
            "interest_pay_date_1",
            "interest_pay_date_2",
        ],
    )
    stocks = monthly_stocks.copy()
    stocks["record_date"] = pd.to_datetime(stocks["record_date"], errors="coerce").dt.normalize()
    stocks = stocks.loc[stocks["record_date"].notna()].copy()
    stocks["outstanding_mil"] = pd.to_numeric(stocks["outstanding_mil"], errors="coerce")
    stocks["coupon_rate_decimal"] = pd.to_numeric(stocks["coupon_rate_decimal"], errors="coerce")
    current = stocks.rename(
        columns={
            "outstanding_mil": "closing_par_mil",
            "security_class": "security_class_current",
            "coupon_rate_decimal": "coupon_rate_decimal_current",
            "maturity_date": "maturity_date_current",
            "interest_pay_date_1": "interest_pay_date_1_current",
            "interest_pay_date_2": "interest_pay_date_2_current",
        }
    )
    prior = stocks.copy()
    prior["record_date"] = prior["record_date"] + pd.offsets.MonthEnd(1)
    prior = prior.rename(
        columns={
            "outstanding_mil": "opening_par_mil",
            "security_class": "security_class_prior",
            "coupon_rate_decimal": "coupon_rate_decimal_prior",
            "maturity_date": "maturity_date_prior",
            "interest_pay_date_1": "interest_pay_date_1_prior",
            "interest_pay_date_2": "interest_pay_date_2_prior",
        }
    )
    panel = current.merge(
        prior[
            [
                "record_date",
                "cusip",
                "opening_par_mil",
                "security_class_prior",
                "coupon_rate_decimal_prior",
                "maturity_date_prior",
                "interest_pay_date_1_prior",
                "interest_pay_date_2_prior",
            ]
        ],
        on=["record_date", "cusip"],
        how="outer",
        sort=False,
    )
    window_start = _quarter_start(start_quarter) if start_quarter else panel["record_date"].min().to_period("Q").start_time.normalize()
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else (panel["record_date"].max().to_period("Q") + 1).start_time.normalize()
    panel = panel.loc[(panel["record_date"] >= window_start) & (panel["record_date"] < window_end)].copy()
    if panel.empty:
        return pd.DataFrame(columns=FIXED_COUPON_MONTHLY_DETAIL_COLUMNS)
    panel["opening_par_mil"] = pd.to_numeric(panel["opening_par_mil"], errors="coerce").fillna(0.0)
    panel["closing_par_mil"] = pd.to_numeric(panel["closing_par_mil"], errors="coerce").fillna(0.0)
    panel["security_class"] = panel["security_class_current"].where(
        panel["security_class_current"].notna(),
        panel["security_class_prior"],
    )
    panel["coupon_rate_decimal"] = panel["coupon_rate_decimal_current"].where(
        panel["coupon_rate_decimal_current"].notna(),
        panel["coupon_rate_decimal_prior"],
    )
    panel["maturity_date"] = panel["maturity_date_current"].where(
        panel["maturity_date_current"].notna(),
        panel["maturity_date_prior"],
    )
    panel["interest_pay_date_1"] = panel["interest_pay_date_1_current"].where(
        panel["interest_pay_date_1_current"].notna(),
        panel["interest_pay_date_1_prior"],
    )
    panel["interest_pay_date_2"] = panel["interest_pay_date_2_current"].where(
        panel["interest_pay_date_2_current"].notna(),
        panel["interest_pay_date_2_prior"],
    )

    events = pd.DataFrame() if auction_events is None else auction_events.copy()
    if not events.empty:
        _require_columns(events, ["cusip", "issue_date", "total_accepted_mil"])
        events["issue_date"] = pd.to_datetime(events["issue_date"], errors="coerce").dt.normalize()
        if "called_date" in events.columns:
            events["called_date"] = pd.to_datetime(events["called_date"], errors="coerce").dt.normalize()
        else:
            events["called_date"] = pd.NaT
        events["total_accepted_mil"] = pd.to_numeric(events["total_accepted_mil"], errors="coerce")
    events_by_cusip = {
        str(cusip).strip(): group.copy()
        for cusip, group in events.groupby("cusip", sort=False, dropna=False)
    } if not events.empty and "cusip" in events.columns else {}
    call_dates_by_cusip = {
        cusip: group["called_date"].dropna().tolist()
        for cusip, group in events_by_cusip.items()
        if "called_date" in group.columns
    }

    rows: list[dict[str, object]] = []
    for _, row in panel.iterrows():
        month_end = pd.to_datetime(row["record_date"], errors="coerce").normalize()
        prior_end = month_end - pd.offsets.MonthEnd(1)
        coupon = _num(row.get("coupon_rate_decimal"), default=np.nan)
        maturity_date = pd.to_datetime(row.get("maturity_date"), errors="coerce")
        if pd.isna(month_end) or pd.isna(coupon) or pd.isna(maturity_date):
            continue
        pay_dates = _coupon_anchor_dates(
            row.get("interest_pay_date_1"),
            row.get("interest_pay_date_2"),
            start=prior_end,
            end=month_end,
        )
        if len(pay_dates) < 2:
            continue
        full_month_factor = _coupon_factor(prior_end, month_end, pay_dates)
        cusip = str(row.get("cusip", "")).strip()
        cusip_events = events_by_cusip.get(cusip, pd.DataFrame())
        event_rows = (
            cusip_events.loc[
                (cusip_events["issue_date"] > prior_end)
                & (cusip_events["issue_date"] <= month_end)
            ]
            if not cusip_events.empty
            else pd.DataFrame()
        )
        known_issuance_mil = float(event_rows["total_accepted_mil"].sum(min_count=1)) if not event_rows.empty else 0.0
        issuance_interest_mil = 0.0
        for _, event in event_rows.iterrows():
            issue_date = pd.to_datetime(event.get("issue_date"), errors="coerce")
            if pd.isna(issue_date):
                continue
            issuance_interest_mil += (
                _num(event.get("total_accepted_mil"), default=0.0)
                * coupon
                * _coupon_factor(issue_date.normalize(), month_end, pay_dates)
            )

        called_dates = call_dates_by_cusip.get(cusip, [])
        termination_date = min(called_dates) if called_dates else maturity_date.normalize()
        opening_par = _num(row.get("opening_par_mil"), default=0.0)
        closing_par = _num(row.get("closing_par_mil"), default=0.0)
        known_redemption_mil = opening_par if prior_end < termination_date <= month_end else 0.0
        redemption_interest_mil = 0.0
        if known_redemption_mil:
            redemption_interest_mil = (
                -known_redemption_mil
                * coupon
                * _coupon_factor(termination_date, month_end, pay_dates)
            )
        residual = closing_par - opening_par - known_issuance_mil + known_redemption_mil
        residual_start_interest = residual * coupon * full_month_factor
        residual_mid_date = _calendar_midpoint(prior_end, month_end)
        residual_mid_interest = residual * coupon * _coupon_factor(residual_mid_date, month_end, pay_dates)
        residual_end_interest = 0.0
        residual_low = min(residual_start_interest, residual_end_interest)
        residual_high = max(residual_start_interest, residual_end_interest)
        opening_interest_mil = opening_par * coupon * full_month_factor
        modeled_interest_mil = (
            opening_interest_mil
            + issuance_interest_mil
            + redemption_interest_mil
            + residual_mid_interest
        )
        rows.append(
            {
                "record_date": month_end.strftime("%Y-%m-%d"),
                "quarter": str(month_end.to_period("Q")),
                "cusip": cusip,
                "security_class": row.get("security_class"),
                "opening_par_mil": opening_par,
                "closing_par_mil": closing_par,
                "coupon_rate_decimal": coupon,
                "maturity_date": maturity_date.strftime("%Y-%m-%d"),
                "interest_pay_date_1": row.get("interest_pay_date_1"),
                "interest_pay_date_2": row.get("interest_pay_date_2"),
                "full_month_coupon_factor": full_month_factor,
                "known_issuance_mil": known_issuance_mil,
                "known_redemption_mil": known_redemption_mil,
                "issuance_interest_mil": issuance_interest_mil,
                "redemption_interest_mil": redemption_interest_mil,
                "unexplained_principal_change_mil": residual,
                "residual_timing_assumption": "calendar_midpoint",
                "residual_interest_low_mil": residual_low,
                "residual_interest_mid_mil": residual_mid_interest,
                "residual_interest_high_mil": residual_high,
                "modeled_interest_mil": modeled_interest_mil,
                "source_coverage_status": "mspd_monthly_stock_terms_and_auction_issuance_observed",
                "model_version": FIXED_COUPON_FLOW_MODEL_VERSION,
                "canonical_status": "candidate_aggregate_fixed_coupon",
            }
        )
    if not rows:
        return pd.DataFrame(columns=FIXED_COUPON_MONTHLY_DETAIL_COLUMNS)
    return (
        pd.DataFrame(rows, columns=FIXED_COUPON_MONTHLY_DETAIL_COLUMNS)
        .sort_values(["record_date", "security_class", "cusip"], kind="stable")
        .reset_index(drop=True)
    )


def build_fixed_coupon_principal_adjustments(detail: pd.DataFrame) -> pd.DataFrame:
    """Emit explicit monthly fixed-coupon principal events and residuals."""

    if detail is None or detail.empty:
        return pd.DataFrame(columns=FIXED_COUPON_PRINCIPAL_ADJUSTMENT_COLUMNS)
    _require_columns(
        detail,
        [
            "record_date",
            "quarter",
            "cusip",
            "security_class",
            "known_issuance_mil",
            "known_redemption_mil",
            "unexplained_principal_change_mil",
            "maturity_date",
        ],
    )
    rows: list[dict[str, object]] = []
    for _, row in detail.iterrows():
        record_date = pd.to_datetime(row.get("record_date"), errors="coerce")
        prior_end = record_date - pd.offsets.MonthEnd(1) if pd.notna(record_date) else pd.NaT
        midpoint = _calendar_midpoint(prior_end, record_date) if pd.notna(prior_end) else pd.NaT
        for event_type, amount, event_date, source in [
            ("issuance", _num(row.get("known_issuance_mil"), default=0.0), record_date, "auction_total_accepted_issue_month"),
            ("redemption", _num(row.get("known_redemption_mil"), default=0.0), row.get("maturity_date"), "mspd_opening_balance_maturity_or_call"),
            (
                "residual",
                _num(row.get("unexplained_principal_change_mil"), default=0.0),
                midpoint,
                "mspd_endpoint_identity_residual_midpoint",
            ),
        ]:
            if abs(amount) <= TGA_FLOOR_TOLERANCE:
                continue
            event_ts = pd.to_datetime(event_date, errors="coerce")
            rows.append(
                {
                    "record_date": row.get("record_date"),
                    "quarter": row.get("quarter"),
                    "cusip": row.get("cusip"),
                    "security_class": row.get("security_class"),
                    "event_type": event_type,
                    "event_date": event_ts.strftime("%Y-%m-%d") if pd.notna(event_ts) else pd.NA,
                    "principal_mil": amount,
                    "source": source,
                    "model_version": row.get("model_version", FIXED_COUPON_FLOW_MODEL_VERSION),
                }
            )
    if not rows:
        return pd.DataFrame(columns=FIXED_COUPON_PRINCIPAL_ADJUSTMENT_COLUMNS)
    return (
        pd.DataFrame(rows, columns=FIXED_COUPON_PRINCIPAL_ADJUSTMENT_COLUMNS)
        .sort_values(["record_date", "cusip", "event_type"], kind="stable")
        .reset_index(drop=True)
    )


def build_fixed_coupon_interest_reconciliation(
    treasury_interest_expense: pd.DataFrame,
    fixed_coupon_flow_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Compare fixed-coupon flow detail to official nominal Notes/Bonds accrued interest."""

    if fixed_coupon_flow_detail is None or fixed_coupon_flow_detail.empty:
        return pd.DataFrame(columns=FIXED_COUPON_INTEREST_RECONCILIATION_COLUMNS)
    _require_columns(
        fixed_coupon_flow_detail,
        [
            "quarter",
            "security_class",
            "modeled_interest_mil",
            "residual_interest_low_mil",
            "residual_interest_high_mil",
        ],
    )
    official = _official_fixed_coupon_by_class(treasury_interest_expense)
    flow = (
        fixed_coupon_flow_detail.groupby(["quarter", "security_class"], dropna=False, sort=False)
        .agg(
            modeled_interest_mil=("modeled_interest_mil", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
            residual_timing_low_mil=("residual_interest_low_mil", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
            residual_timing_high_mil=("residual_interest_high_mil", lambda values: pd.to_numeric(values, errors="coerce").sum(min_count=1)),
        )
        .reset_index()
    )
    combined = (
        flow.groupby("quarter", dropna=False, sort=False)
        .agg(
            modeled_interest_mil=("modeled_interest_mil", "sum"),
            residual_timing_low_mil=("residual_timing_low_mil", "sum"),
            residual_timing_high_mil=("residual_timing_high_mil", "sum"),
        )
        .reset_index()
    )
    combined["security_class"] = "Notes+Bonds"
    flow = pd.concat([flow, combined], ignore_index=True)
    merged = official.merge(flow, on=["quarter", "security_class"], how="outer")
    merged["gap_mil"] = (
        pd.to_numeric(merged["modeled_interest_mil"], errors="coerce")
        - pd.to_numeric(merged["official_interest_mil"], errors="coerce")
    )
    merged["abs_gap_mil"] = merged["gap_mil"].abs()
    merged["gap_pct"] = merged["gap_mil"] / pd.to_numeric(merged["official_interest_mil"], errors="coerce") * 100.0
    merged["abs_gap_pct"] = merged["gap_pct"].abs()
    merged["residual_timing_envelope_width_mil"] = (
        pd.to_numeric(merged["residual_timing_high_mil"], errors="coerce")
        - pd.to_numeric(merged["residual_timing_low_mil"], errors="coerce")
    )
    merged["source_coverage_pct"] = 100.0
    merged["model_version"] = FIXED_COUPON_FLOW_MODEL_VERSION
    merged["canonical_status"] = "candidate_aggregate_fixed_coupon"
    return (
        merged.loc[:, FIXED_COUPON_INTEREST_RECONCILIATION_COLUMNS]
        .sort_values(["quarter", "security_class"], kind="stable")
        .reset_index(drop=True)
    )


def build_frn_daily_index_path(
    benchmark_auctions: pd.DataFrame,
    frn_auction_lots: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build per-CUSIP FRN benchmark index paths from 13-week bill auctions."""

    if benchmark_auctions is None or benchmark_auctions.empty or frn_auction_lots is None or frn_auction_lots.empty:
        return pd.DataFrame(columns=FRN_DAILY_INDEX_PATH_COLUMNS)
    _require_columns(benchmark_auctions, ["auction_date", "effective_date", "benchmark_index_pct", "term_days"])
    _require_columns(
        frn_auction_lots,
        [
            "cusip",
            "lot_id",
            "issue_date",
            "maturity_date",
            "frn_index_determination_rate",
            "frn_index_determination_date",
            "is_reopening",
        ],
    )
    benchmark = benchmark_auctions.copy()
    benchmark["effective_date"] = pd.to_datetime(benchmark["effective_date"], errors="coerce").dt.normalize()
    benchmark["auction_date"] = pd.to_datetime(benchmark["auction_date"], errors="coerce").dt.normalize()
    benchmark["benchmark_index_pct"] = pd.to_numeric(benchmark["benchmark_index_pct"], errors="coerce")
    benchmark["term_days"] = pd.to_numeric(benchmark["term_days"], errors="coerce")
    benchmark = benchmark.dropna(subset=["effective_date", "benchmark_index_pct"]).sort_values(
        ["effective_date", "auction_date"],
        kind="stable",
    )
    term_days_by_auction = benchmark.set_index("auction_date")["term_days"].to_dict()
    lots = frn_auction_lots.copy()
    lots["issue_date"] = pd.to_datetime(lots["issue_date"], errors="coerce").dt.normalize()
    lots["maturity_date"] = pd.to_datetime(lots["maturity_date"], errors="coerce").dt.normalize()
    lots["frn_index_determination_date"] = pd.to_datetime(
        lots["frn_index_determination_date"],
        errors="coerce",
    ).dt.normalize()
    lots["frn_index_determination_rate"] = pd.to_numeric(
        lots["frn_index_determination_rate"],
        errors="coerce",
    )
    lots["is_reopening"] = lots["is_reopening"].fillna(False).astype(bool)
    lots = lots.dropna(subset=["issue_date", "maturity_date"])
    if lots.empty:
        return pd.DataFrame(columns=FRN_DAILY_INDEX_PATH_COLUMNS)
    window_start = _quarter_start(start_quarter) if start_quarter else lots["issue_date"].min()
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else lots["maturity_date"].max()
    active_lots = lots.loc[(lots["maturity_date"] > window_start) & (lots["issue_date"] < window_end)].copy()
    if active_lots.empty:
        return pd.DataFrame(columns=FRN_DAILY_INDEX_PATH_COLUMNS)
    original_overrides = lots.loc[~lots["is_reopening"]].copy()
    rows: list[pd.DataFrame] = []
    for cusip, group in active_lots.groupby("cusip", sort=False, dropna=False):
        cusip = str(cusip).strip()
        cusip_originals = group.loc[~group["is_reopening"]].copy()
        cusip_lockout_starts = (
            cusip_originals["issue_date"].dropna().map(lambda value: (value - 2 * _US_FEDERAL_BUSINESS_DAY).normalize())
        )
        source_start = min([group["issue_date"].min(), *cusip_lockout_starts.tolist()])
        start = max(window_start, source_start)
        end_exclusive = min(window_end, group["maturity_date"].max())
        if end_exclusive <= start:
            continue
        calendar_frame = pd.DataFrame({"accrual_date": pd.date_range(start, end_exclusive - pd.Timedelta(days=1), freq="D")})
        path = pd.merge_asof(
            calendar_frame.sort_values("accrual_date"),
            benchmark[["effective_date", "benchmark_index_pct"]].sort_values("effective_date"),
            left_on="accrual_date",
            right_on="effective_date",
            direction="backward",
        )
        path["daily_index_pct"] = path["benchmark_index_pct"]
        path["index_source"] = "13_week_bill_high_discount_rate"
        overrides = pd.concat(
            [original_overrides, lots.loc[lots["cusip"].astype(str).str.strip().eq(cusip) & lots["is_reopening"]]],
            ignore_index=True,
        )
        for _, event in overrides.dropna(subset=["issue_date", "frn_index_determination_rate"]).iterrows():
            issue_date = pd.to_datetime(event.get("issue_date"), errors="coerce")
            if pd.isna(issue_date):
                continue
            determination_date = pd.to_datetime(event.get("frn_index_determination_date"), errors="coerce")
            term_days = term_days_by_auction.get(determination_date.normalize() if pd.notna(determination_date) else pd.NaT, 91.0)
            determination_rate = _num(event.get("frn_index_determination_rate"), default=np.nan)
            if pd.isna(determination_rate):
                continue
            override_index = _frn_discount_to_index_pct(determination_rate, term_days)
            lockout_start = (issue_date.normalize() - 2 * _US_FEDERAL_BUSINESS_DAY).normalize()
            lockout_end = issue_date.normalize()
            mask = path["accrual_date"].between(lockout_start, lockout_end)
            path.loc[mask, "daily_index_pct"] = override_index
            path.loc[mask, "index_source"] = (
                "frn_reopening_index_determination_lockout"
                if bool(event.get("is_reopening", False))
                else "frn_original_issue_index_determination_lockout"
            )
        path["cusip"] = cusip
        path["model_version"] = FRN_FLOW_MODEL_VERSION
        rows.append(path.loc[:, FRN_DAILY_INDEX_PATH_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=FRN_DAILY_INDEX_PATH_COLUMNS)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["cusip", "accrual_date"], kind="stable")
        .reset_index(drop=True)
    )


def build_frn_daily_index_backcast_validation(
    daily_index_path: pd.DataFrame,
    observed_daily_indexes: pd.DataFrame,
    *,
    tolerance: float = 1.0e-9,
) -> pd.DataFrame:
    """Validate reconstructed FRN daily index rows against Treasury rows."""

    if daily_index_path is None or daily_index_path.empty or observed_daily_indexes is None or observed_daily_indexes.empty:
        return pd.DataFrame(columns=FRN_DAILY_INDEX_VALIDATION_COLUMNS)
    _require_columns(daily_index_path, ["cusip", "accrual_date", "daily_index_pct"])
    _require_columns(
        observed_daily_indexes,
        [
            "cusip",
            "start_of_accrual_period",
            "daily_index",
            "spread",
            "daily_int_accrual_rate",
            "daily_accrued_int_per100",
        ],
    )
    modeled = daily_index_path.copy()
    modeled["accrual_date"] = pd.to_datetime(modeled["accrual_date"], errors="coerce").dt.normalize()
    modeled["daily_index_pct"] = pd.to_numeric(modeled["daily_index_pct"], errors="coerce")
    observed = observed_daily_indexes.copy()
    observed["accrual_date"] = pd.to_datetime(observed["start_of_accrual_period"], errors="coerce").dt.normalize()
    for column in ["daily_index", "spread", "daily_int_accrual_rate", "daily_accrued_int_per100"]:
        observed[column] = pd.to_numeric(observed[column], errors="coerce")
    merged = observed.merge(
        modeled[["cusip", "accrual_date", "daily_index_pct"]],
        on=["cusip", "accrual_date"],
        how="left",
    )
    merged["observed_daily_index_pct"] = merged["daily_index"]
    merged["modeled_daily_index_pct"] = merged["daily_index_pct"]
    merged["index_diff_pct"] = merged["modeled_daily_index_pct"] - merged["observed_daily_index_pct"]
    merged["observed_spread_pct"] = merged["spread"]
    merged["modeled_spread_pct"] = merged["spread"]
    merged["spread_diff_pct"] = 0.0
    merged["observed_daily_int_accrual_rate_pct"] = merged["daily_int_accrual_rate"]
    merged["modeled_daily_int_accrual_rate_pct"] = (
        merged["modeled_daily_index_pct"] + merged["observed_spread_pct"]
    ).clip(lower=0.0)
    merged["accrual_rate_diff_pct"] = (
        merged["modeled_daily_int_accrual_rate_pct"]
        - merged["observed_daily_int_accrual_rate_pct"]
    )
    merged["observed_daily_accrued_int_per100"] = merged["daily_accrued_int_per100"]
    merged["modeled_daily_accrued_int_per100"] = merged["modeled_daily_int_accrual_rate_pct"] / 360.0
    merged["daily_accrued_int_per100_diff"] = (
        merged["modeled_daily_accrued_int_per100"]
        - merged["observed_daily_accrued_int_per100"]
    )
    merged["validation_pass"] = (
        merged["index_diff_pct"].abs().le(tolerance)
        & merged["accrual_rate_diff_pct"].abs().le(tolerance)
        & merged["daily_accrued_int_per100_diff"].abs().le(tolerance)
    )
    return (
        merged.loc[:, FRN_DAILY_INDEX_VALIDATION_COLUMNS]
        .sort_values(["cusip", "accrual_date"], kind="stable")
        .reset_index(drop=True)
    )


def build_frn_interest_flow_detail(
    frn_auction_lots: pd.DataFrame,
    daily_index_path: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build daily FRN interest from auction principal lots and benchmark path."""

    if frn_auction_lots is None or frn_auction_lots.empty or daily_index_path is None or daily_index_path.empty:
        return pd.DataFrame(columns=FRN_INTEREST_DAILY_DETAIL_COLUMNS)
    _require_columns(
        frn_auction_lots,
        ["lot_id", "cusip", "auction_date", "issue_date", "maturity_date", "total_accepted_mil", "spread_pct", "is_reopening"],
    )
    _require_columns(daily_index_path, ["cusip", "accrual_date", "daily_index_pct"])
    lots = frn_auction_lots.copy()
    lots["issue_date"] = pd.to_datetime(lots["issue_date"], errors="coerce").dt.normalize()
    lots["maturity_date"] = pd.to_datetime(lots["maturity_date"], errors="coerce").dt.normalize()
    lots["total_accepted_mil"] = pd.to_numeric(lots["total_accepted_mil"], errors="coerce")
    lots["spread_pct"] = pd.to_numeric(lots["spread_pct"], errors="coerce")
    lots = lots.dropna(subset=["issue_date", "maturity_date", "total_accepted_mil", "spread_pct"])
    path = daily_index_path.copy()
    path["accrual_date"] = pd.to_datetime(path["accrual_date"], errors="coerce").dt.normalize()
    path["daily_index_pct"] = pd.to_numeric(path["daily_index_pct"], errors="coerce")
    path_by_cusip = {str(cusip).strip(): group.copy() for cusip, group in path.groupby("cusip", sort=False, dropna=False)}
    window_start = _quarter_start(start_quarter) if start_quarter else lots["issue_date"].min()
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else lots["maturity_date"].max()
    rows: list[pd.DataFrame] = []
    for _, lot in lots.iterrows():
        issue_date = pd.to_datetime(lot.get("issue_date"), errors="coerce")
        maturity_date = pd.to_datetime(lot.get("maturity_date"), errors="coerce")
        accrual_start = max(issue_date, window_start)
        accrual_end = min(maturity_date, window_end)
        if pd.isna(accrual_start) or pd.isna(accrual_end) or accrual_end <= accrual_start:
            continue
        cusip = str(lot.get("cusip", "")).strip()
        lot_path = path_by_cusip.get(cusip, pd.DataFrame())
        if lot_path.empty:
            continue
        lot_path = lot_path.loc[
            (lot_path["accrual_date"] >= accrual_start)
            & (lot_path["accrual_date"] < accrual_end)
        ].copy()
        if lot_path.empty:
            continue
        spread_pct = _num(lot.get("spread_pct"), default=0.0)
        par_mil = _num(lot.get("total_accepted_mil"), default=0.0)
        lot_path["quarter"] = lot_path["accrual_date"].dt.to_period("Q").astype(str)
        lot_path["lot_id"] = lot.get("lot_id")
        lot_path["auction_date"] = lot.get("auction_date")
        lot_path["issue_date"] = issue_date.strftime("%Y-%m-%d")
        lot_path["maturity_date"] = maturity_date.strftime("%Y-%m-%d")
        lot_path["is_reopening"] = bool(lot.get("is_reopening", False))
        lot_path["par_mil"] = par_mil
        lot_path["spread_pct"] = spread_pct
        lot_path["daily_int_accrual_rate_pct"] = (lot_path["daily_index_pct"] + spread_pct).clip(lower=0.0)
        lot_path["daily_accrued_int_per100"] = lot_path["daily_int_accrual_rate_pct"] / 360.0
        lot_path["modeled_interest_mil"] = par_mil * lot_path["daily_int_accrual_rate_pct"] / 100.0 / 360.0
        lot_path["source_coverage_status"] = "auction_lot_and_13_week_bill_index_observed"
        lot_path["model_version"] = FRN_FLOW_MODEL_VERSION
        lot_path["canonical_status"] = "candidate_aggregate_frn_interest"
        rows.append(lot_path.loc[:, FRN_INTEREST_DAILY_DETAIL_COLUMNS])
    if not rows:
        return pd.DataFrame(columns=FRN_INTEREST_DAILY_DETAIL_COLUMNS)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["accrual_date", "cusip", "lot_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_frn_principal_reconciliation(
    frn_mspd_principal_controls: pd.DataFrame,
    frn_auction_lots: pd.DataFrame,
) -> pd.DataFrame:
    """Compare FRN auction-lot principal to monthly MSPD controls."""

    if frn_mspd_principal_controls is None or frn_mspd_principal_controls.empty:
        return pd.DataFrame(columns=FRN_PRINCIPAL_RECONCILIATION_COLUMNS)
    if frn_auction_lots is None or frn_auction_lots.empty:
        return pd.DataFrame(columns=FRN_PRINCIPAL_RECONCILIATION_COLUMNS)
    controls = frn_mspd_principal_controls.copy()
    controls["record_date"] = pd.to_datetime(controls["record_date"], errors="coerce").dt.normalize()
    controls["control_outstanding_mil"] = pd.to_numeric(controls["control_outstanding_mil"], errors="coerce")
    lots = frn_auction_lots.copy()
    lots["issue_date"] = pd.to_datetime(lots["issue_date"], errors="coerce").dt.normalize()
    lots["maturity_date"] = pd.to_datetime(lots["maturity_date"], errors="coerce").dt.normalize()
    lots["total_accepted_mil"] = pd.to_numeric(lots["total_accepted_mil"], errors="coerce")
    rows: list[dict[str, object]] = []
    lots_by_cusip = {str(cusip).strip(): group.copy() for cusip, group in lots.groupby("cusip", sort=False, dropna=False)}
    for _, row in controls.iterrows():
        record_date = pd.to_datetime(row.get("record_date"), errors="coerce")
        cusip = str(row.get("cusip", "")).strip()
        group = lots_by_cusip.get(cusip, pd.DataFrame())
        auction_outstanding = np.nan
        if not group.empty and pd.notna(record_date):
            active = group.loc[(group["issue_date"] <= record_date) & (group["maturity_date"] > record_date)]
            auction_outstanding = pd.to_numeric(active["total_accepted_mil"], errors="coerce").sum(min_count=1)
        control = _num(row.get("control_outstanding_mil"), default=np.nan)
        diff = auction_outstanding - control if pd.notna(auction_outstanding) and pd.notna(control) else np.nan
        rows.append(
            {
                "record_date": record_date.strftime("%Y-%m-%d") if pd.notna(record_date) else pd.NA,
                "quarter": row.get("quarter"),
                "cusip": cusip,
                "mspd_control_outstanding_mil": control,
                "auction_lot_outstanding_mil": auction_outstanding,
                "difference_mil": diff,
                "abs_difference_mil": abs(diff) if pd.notna(diff) else np.nan,
                "difference_pct": diff / control * 100.0
                if pd.notna(diff) and pd.notna(control) and abs(control) > TGA_FLOOR_TOLERANCE
                else np.nan,
                "source_coverage_status": "auction_lot_principal_compared_to_mspd_monthly_control",
                "model_version": FRN_FLOW_MODEL_VERSION,
            }
        )
    return (
        pd.DataFrame(rows, columns=FRN_PRINCIPAL_RECONCILIATION_COLUMNS)
        .sort_values(["record_date", "cusip"], kind="stable")
        .reset_index(drop=True)
    )


def build_frn_cusip_coverage(
    frn_auction_lots: pd.DataFrame,
    daily_index_path: pd.DataFrame,
    observed_daily_indexes: pd.DataFrame,
    frn_mspd_principal_controls: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize FRN source coverage by CUSIP for audit checks."""

    if frn_auction_lots is None or frn_auction_lots.empty:
        return pd.DataFrame(columns=FRN_CUSIP_COVERAGE_COLUMNS)
    _require_columns(frn_auction_lots, ["cusip", "issue_date", "maturity_date", "total_accepted_mil"])
    lots = frn_auction_lots.copy()
    lots["cusip"] = lots["cusip"].astype(str).str.strip()
    lots["issue_date"] = pd.to_datetime(lots["issue_date"], errors="coerce").dt.normalize()
    lots["maturity_date"] = pd.to_datetime(lots["maturity_date"], errors="coerce").dt.normalize()
    lots["total_accepted_mil"] = pd.to_numeric(lots["total_accepted_mil"], errors="coerce")
    coverage = (
        lots.groupby("cusip", sort=False, dropna=False)
        .agg(
            first_issue_date=("issue_date", "min"),
            maturity_date=("maturity_date", "max"),
            auction_lot_count=("cusip", "size"),
            auction_lot_par_mil=("total_accepted_mil", "sum"),
        )
        .reset_index()
    )
    modeled_counts = _frn_count_by_cusip(daily_index_path, "accrual_date", "modeled_daily_rows")
    observed_counts = _frn_count_by_cusip(observed_daily_indexes, "start_of_accrual_period", "observed_daily_rows")
    control_counts = _frn_count_by_cusip(frn_mspd_principal_controls, "record_date", "mspd_control_rows")
    coverage = coverage.merge(modeled_counts, on="cusip", how="left")
    coverage = coverage.merge(observed_counts, on="cusip", how="left")
    coverage = coverage.merge(control_counts, on="cusip", how="left")
    for column in ["modeled_daily_rows", "observed_daily_rows", "mspd_control_rows"]:
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0).astype(int)
    coverage["coverage_status"] = np.select(
        [
            coverage["modeled_daily_rows"].gt(0)
            & coverage["observed_daily_rows"].gt(0)
            & coverage["mspd_control_rows"].gt(0),
            coverage["modeled_daily_rows"].gt(0) & coverage["mspd_control_rows"].gt(0),
        ],
        [
            "auction_lot_modeled_daily_observed_daily_mspd_control",
            "auction_lot_modeled_daily_mspd_control",
        ],
        default="coverage_incomplete",
    )
    coverage["first_issue_date"] = coverage["first_issue_date"].dt.strftime("%Y-%m-%d")
    coverage["maturity_date"] = coverage["maturity_date"].dt.strftime("%Y-%m-%d")
    coverage["model_version"] = FRN_FLOW_MODEL_VERSION
    return (
        coverage.loc[:, FRN_CUSIP_COVERAGE_COLUMNS]
        .sort_values(["first_issue_date", "cusip"], kind="stable")
        .reset_index(drop=True)
    )


def build_frn_interest_reconciliation(
    treasury_interest_expense: pd.DataFrame,
    frn_interest_flow_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Compare FRN auction-lot modeled interest to official FRN accrued interest."""

    if frn_interest_flow_detail is None or frn_interest_flow_detail.empty:
        return pd.DataFrame(columns=FRN_INTEREST_RECONCILIATION_COLUMNS)
    official = _official_frn_interest_by_quarter(treasury_interest_expense)
    flow = frn_interest_flow_detail.copy()
    _require_columns(flow, ["quarter", "modeled_interest_mil"])
    flow["modeled_interest_mil"] = pd.to_numeric(flow["modeled_interest_mil"], errors="coerce")
    modeled = (
        flow.groupby("quarter", sort=False, dropna=False)["modeled_interest_mil"]
        .sum(min_count=1)
        .reset_index()
    )
    merged = official.merge(modeled, on="quarter", how="outer")
    merged["gap_mil"] = pd.to_numeric(merged["modeled_interest_mil"], errors="coerce") - pd.to_numeric(
        merged["official_interest_mil"],
        errors="coerce",
    )
    merged["abs_gap_mil"] = merged["gap_mil"].abs()
    merged["gap_pct"] = merged["gap_mil"] / pd.to_numeric(merged["official_interest_mil"], errors="coerce") * 100.0
    merged["abs_gap_pct"] = merged["gap_pct"].abs()
    merged["source_coverage_pct"] = 100.0
    merged["model_version"] = FRN_FLOW_MODEL_VERSION
    merged["canonical_status"] = "candidate_aggregate_frn_interest"
    return (
        merged.loc[:, FRN_INTEREST_RECONCILIATION_COLUMNS]
        .sort_values("quarter", kind="stable")
        .reset_index(drop=True)
    )


def build_tips_inflation_compensation_flow_detail(
    tips_inflation_adjustment_stocks: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
    tips_auction_events: pd.DataFrame | None = None,
    tips_cpi_reference_path: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build TIPS inflation compensation from MSPD CUSIP stocks and source events."""

    if tips_inflation_adjustment_stocks is None or tips_inflation_adjustment_stocks.empty:
        return pd.DataFrame(columns=TIPS_INFLATION_MONTHLY_DETAIL_COLUMNS)
    _require_columns(
        tips_inflation_adjustment_stocks,
        [
            "record_date",
            "quarter",
            "cohort_id",
            "cusip",
            "issue_date",
            "maturity_date",
            "issued_amt_mil",
            "inflation_adjustment_mil",
            "redeemed_amt_mil",
        ],
    )
    stocks = tips_inflation_adjustment_stocks.copy()
    stocks["record_date"] = pd.to_datetime(stocks["record_date"], errors="coerce").dt.normalize()
    stocks["issue_date"] = pd.to_datetime(stocks["issue_date"], errors="coerce").dt.normalize()
    stocks["maturity_date"] = pd.to_datetime(stocks["maturity_date"], errors="coerce").dt.normalize()
    stocks["cusip"] = stocks["cusip"].astype(str).str.strip().str.upper()
    for column in ["inflation_adjustment_mil", "issued_amt_mil", "redeemed_amt_mil"]:
        stocks[column] = pd.to_numeric(stocks[column], errors="coerce")
    stocks["redeemed_amt_mil"] = stocks["redeemed_amt_mil"].fillna(0.0)
    stocks = stocks.dropna(subset=["record_date", "cusip", "inflation_adjustment_mil"])
    if stocks.empty:
        return pd.DataFrame(columns=TIPS_INFLATION_MONTHLY_DETAIL_COLUMNS)
    stock_frame = (
        stocks.groupby(["cusip", "record_date"], sort=False, dropna=False)
        .agg(
            issue_date=("issue_date", "min"),
            maturity_date=("maturity_date", "max"),
            issued_amt_mil=("issued_amt_mil", "sum"),
            inflation_adjustment_mil=("inflation_adjustment_mil", "sum"),
            redeemed_amt_mil=("redeemed_amt_mil", "sum"),
        )
        .reset_index()
    )
    stock_frame["cohort_id"] = stock_frame["cusip"]
    stock_frame = stock_frame.sort_values(["cusip", "record_date"], kind="stable").reset_index(drop=True)
    by_cusip = stock_frame.groupby("cusip", sort=False)
    month_ordinal = stock_frame["record_date"].dt.to_period("M").astype("int64")
    prior_month_ordinal = month_ordinal.groupby(stock_frame["cusip"], sort=False).shift(1)
    prior_record_date = by_cusip["record_date"].shift(1)
    modern_clean_vintage = stock_frame["record_date"].ge(pd.Timestamp("2007-01-01")) & prior_record_date.ge(
        pd.Timestamp("2007-01-01")
    )
    missing_month = (
        prior_month_ordinal.notna()
        & month_ordinal.sub(prior_month_ordinal).ne(1)
        & modern_clean_vintage
    )
    if missing_month.any():
        sample = stock_frame.loc[missing_month, ["cusip", "record_date"]].head(5).to_dict("records")
        raise ValueError(f"tips inflation adjustment stocks have non-monthly gaps: {sample}")
    stock_frame["prior_inflation_adjustment_mil"] = by_cusip["inflation_adjustment_mil"].shift(1)
    stock_frame["prior_redeemed_amt_mil"] = by_cusip["redeemed_amt_mil"].shift(1).fillna(0.0)
    stock_frame["is_first_observed_cohort_month"] = stock_frame["prior_inflation_adjustment_mil"].isna()
    raw_stock_delta = (
        stock_frame["inflation_adjustment_mil"] - stock_frame["prior_inflation_adjustment_mil"]
    )
    raw_stock_delta = raw_stock_delta.where(~stock_frame["is_first_observed_cohort_month"], stock_frame["inflation_adjustment_mil"])
    stock_frame["redeemed_delta_mil"] = stock_frame["redeemed_amt_mil"] - stock_frame["prior_redeemed_amt_mil"]
    stock_frame["redeemed_increment_mil"] = (-stock_frame["redeemed_delta_mil"]).clip(lower=0.0)
    original_par_after_redemption = stock_frame["issued_amt_mil"] + stock_frame["redeemed_amt_mil"]
    full_redemption_gap = stock_frame["redeemed_increment_mil"].gt(0.0) & original_par_after_redemption.le(0.0)
    full_redemption_gap &= stock_frame["record_date"].ge(pd.Timestamp("2010-01-01"))
    if full_redemption_gap.any():
        sample = stock_frame.loc[full_redemption_gap, ["cusip", "record_date", "issued_amt_mil", "redeemed_amt_mil"]].head(5).to_dict("records")
        raise ValueError(f"tips inflation adjustment stocks have full-redemption rows requiring event timing: {sample}")
    redemption_ratio = stock_frame["inflation_adjustment_mil"] / original_par_after_redemption.where(
        original_par_after_redemption.gt(0.0)
    )
    stock_frame["redemption_adjustment_mil"] = (
        stock_frame["redeemed_increment_mil"] * redemption_ratio
    ).fillna(0.0)
    stock_frame.loc[stock_frame["is_first_observed_cohort_month"], "redeemed_delta_mil"] = 0.0
    stock_frame.loc[stock_frame["is_first_observed_cohort_month"], "redeemed_increment_mil"] = 0.0
    stock_frame.loc[stock_frame["is_first_observed_cohort_month"], "redemption_adjustment_mil"] = 0.0

    stock_frame["issue_adjustment_mil"] = _tips_issue_adjustments_by_cusip_month(
        tips_auction_events,
    ).reindex(
        pd.MultiIndex.from_frame(
            pd.DataFrame(
                {
                    "cusip": stock_frame["cusip"],
                    "record_month": stock_frame["record_date"].dt.to_period("M").astype(str),
                }
            )
        )
    ).fillna(0.0).to_numpy()
    stock_frame["stock_delta_inflation_adjustment_mil"] = raw_stock_delta
    opening_without_issue = stock_frame["is_first_observed_cohort_month"] & stock_frame["issue_adjustment_mil"].le(0.0)
    stock_frame.loc[opening_without_issue, "stock_delta_inflation_adjustment_mil"] = 0.0
    stock_frame["maturity_floor_adjustment_mil"] = 0.0
    stock_frame["modeled_inflation_compensation_mil"] = (
        stock_frame["stock_delta_inflation_adjustment_mil"]
        - stock_frame["issue_adjustment_mil"]
        + stock_frame["redemption_adjustment_mil"]
    )
    stock_frame["event_type"] = "mspd_cusip_stock_flow"
    stock_frame["source_coverage_status"] = (
        "mspd_cusip_inflation_adjustment_stock_flow_with_issue_redemption_adjustments"
    )
    stock_frame["model_version"] = TIPS_INFLATION_FLOW_MODEL_VERSION
    stock_frame["canonical_status"] = "candidate_aggregate_tips_inflation_compensation_event_flow"

    maturity_frame = _tips_maturity_inflation_events(
        stock_frame,
        tips_auction_events,
        tips_cpi_reference_path,
        start_quarter=start_quarter,
        end_quarter=end_quarter,
    )
    frame = pd.concat([stock_frame, maturity_frame], ignore_index=True, sort=False)
    frame["record_date"] = pd.to_datetime(frame["record_date"], errors="coerce").dt.normalize()
    frame["issue_date"] = pd.to_datetime(frame["issue_date"], errors="coerce").dt.normalize()
    frame["maturity_date"] = pd.to_datetime(frame["maturity_date"], errors="coerce").dt.normalize()
    window_start = _quarter_start(start_quarter) if start_quarter else None
    if window_start is not None:
        frame = frame.loc[frame["record_date"] >= window_start].copy()
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else None
    if window_end is not None:
        frame = frame.loc[frame["record_date"] < window_end].copy()
    if frame.empty:
        return pd.DataFrame(columns=TIPS_INFLATION_MONTHLY_DETAIL_COLUMNS)
    frame["quarter"] = frame["record_date"].dt.to_period("Q").astype(str)
    frame["record_date"] = frame["record_date"].dt.strftime("%Y-%m-%d")
    frame["issue_date"] = frame["issue_date"].dt.strftime("%Y-%m-%d")
    frame["maturity_date"] = frame["maturity_date"].dt.strftime("%Y-%m-%d")
    return (
        frame.loc[:, TIPS_INFLATION_MONTHLY_DETAIL_COLUMNS]
        .sort_values(["record_date", "event_type", "cohort_id"], kind="stable")
        .reset_index(drop=True)
    )


def _tips_issue_adjustments_by_cusip_month(tips_auction_events: pd.DataFrame | None) -> pd.Series:
    if tips_auction_events is None or tips_auction_events.empty:
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["cusip", "record_month"]))
    required = {"cusip", "issue_date", "total_accepted_mil"}
    if not required.issubset(tips_auction_events.columns):
        return pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[], []], names=["cusip", "record_month"]))
    events = tips_auction_events.copy()
    events["cusip"] = events["cusip"].astype(str).str.strip().str.upper()
    events["issue_date"] = pd.to_datetime(events["issue_date"], errors="coerce").dt.normalize()
    events["total_accepted_mil"] = pd.to_numeric(events["total_accepted_mil"], errors="coerce")
    if "index_ratio_on_issue_date" in events.columns:
        index_ratio = pd.to_numeric(events["index_ratio_on_issue_date"], errors="coerce")
    else:
        index_ratio = pd.Series(np.nan, index=events.index)
    if "ref_cpi_on_issue_date" in events.columns and "ref_cpi_on_dated_date" in events.columns:
        issue_cpi = pd.to_numeric(events["ref_cpi_on_issue_date"], errors="coerce")
        base_cpi = pd.to_numeric(events["ref_cpi_on_dated_date"], errors="coerce")
        derived_ratio = issue_cpi / base_cpi.where(base_cpi.gt(0.0))
        index_ratio = index_ratio.where(index_ratio.notna(), derived_ratio)
    events["issue_adjustment_mil"] = events["total_accepted_mil"] * (index_ratio - 1.0).clip(lower=0.0)
    events["record_month"] = events["issue_date"].dt.to_period("M").astype(str)
    grouped = (
        events.dropna(subset=["cusip", "record_month"])
        .groupby(["cusip", "record_month"], sort=False)["issue_adjustment_mil"]
        .sum(min_count=1)
    )
    grouped.index = grouped.index.set_names(["cusip", "record_month"])
    return grouped


def _tips_maturity_inflation_events(
    stock_frame: pd.DataFrame,
    tips_auction_events: pd.DataFrame | None,
    tips_cpi_reference_path: pd.DataFrame | None,
    *,
    start_quarter: str | None,
    end_quarter: str | None,
) -> pd.DataFrame:
    if (
        stock_frame.empty
        or tips_auction_events is None
        or tips_auction_events.empty
        or tips_cpi_reference_path is None
        or tips_cpi_reference_path.empty
    ):
        return pd.DataFrame(columns=stock_frame.columns)
    events = tips_auction_events.copy()
    if not {"cusip", "ref_cpi_on_dated_date"}.issubset(events.columns):
        return pd.DataFrame(columns=stock_frame.columns)
    events["cusip"] = events["cusip"].astype(str).str.strip().str.upper()
    events["ref_cpi_on_dated_date"] = pd.to_numeric(events["ref_cpi_on_dated_date"], errors="coerce")
    base_cpi = (
        events.dropna(subset=["cusip", "ref_cpi_on_dated_date"])
        .sort_values(["cusip", "issue_date"], kind="stable")
        .groupby("cusip", sort=False)["ref_cpi_on_dated_date"]
        .first()
    )
    cpi = tips_cpi_reference_path.copy()
    cpi["index_date"] = pd.to_datetime(cpi["index_date"], errors="coerce").dt.normalize()
    cpi["ref_cpi"] = pd.to_numeric(cpi["ref_cpi"], errors="coerce")
    cpi_by_date = cpi.dropna(subset=["index_date", "ref_cpi"]).drop_duplicates("index_date").set_index("index_date")["ref_cpi"]
    window_start = _quarter_start(start_quarter) if start_quarter else pd.Timestamp.min
    window_end = _quarter_end_exclusive(end_quarter) if end_quarter else pd.Timestamp.max
    rows: list[dict[str, object]] = []
    last_by_cusip = stock_frame.sort_values(["cusip", "record_date"], kind="stable").groupby("cusip", sort=False).tail(1)
    for _, row in last_by_cusip.iterrows():
        maturity_date = row.get("maturity_date")
        if pd.isna(maturity_date) or maturity_date < window_start or maturity_date >= window_end:
            continue
        if row["record_date"] >= maturity_date:
            continue
        cusip = str(row["cusip"]).strip().upper()
        dated_cpi = base_cpi.get(cusip, np.nan)
        maturity_cpi = cpi_by_date.get(maturity_date, np.nan)
        remaining_par = _num(row.get("issued_amt_mil"), default=0.0) + _num(row.get("redeemed_amt_mil"), default=0.0)
        last_adjustment = _num(row.get("inflation_adjustment_mil"), default=0.0)
        if pd.isna(dated_cpi) or pd.isna(maturity_cpi) or dated_cpi <= 0.0 or remaining_par <= 0.0:
            continue
        maturity_ratio = float(maturity_cpi) / float(dated_cpi)
        floor_adjustment = remaining_par * max(maturity_ratio - 1.0, 0.0)
        event = row.to_dict()
        event.update(
            {
                "record_date": maturity_date,
                "event_type": "synthetic_maturity_floor_event",
                "cohort_id": f"{cusip}|maturity_event",
                "prior_inflation_adjustment_mil": last_adjustment,
                "stock_delta_inflation_adjustment_mil": -last_adjustment,
                "issue_adjustment_mil": 0.0,
                "prior_redeemed_amt_mil": row.get("redeemed_amt_mil", 0.0),
                "redeemed_delta_mil": 0.0,
                "redeemed_increment_mil": 0.0,
                "redemption_adjustment_mil": 0.0,
                "maturity_floor_adjustment_mil": floor_adjustment,
                "modeled_inflation_compensation_mil": -last_adjustment + floor_adjustment,
                "is_first_observed_cohort_month": False,
                "source_coverage_status": "mspd_cusip_stock_flow_with_tips_cpi_maturity_floor_event",
                "model_version": TIPS_INFLATION_FLOW_MODEL_VERSION,
                "canonical_status": "candidate_aggregate_tips_inflation_compensation_event_flow",
            }
        )
        rows.append(event)
    if not rows:
        return pd.DataFrame(columns=stock_frame.columns)
    return pd.DataFrame(rows)


def build_tips_inflation_compensation_reconciliation(
    treasury_interest_expense: pd.DataFrame,
    tips_inflation_flow_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Compare modeled TIPS inflation compensation to official public-issue pool."""

    if tips_inflation_flow_detail is None or tips_inflation_flow_detail.empty:
        return pd.DataFrame(columns=TIPS_INFLATION_RECONCILIATION_COLUMNS)
    official = _official_tips_inflation_by_quarter(treasury_interest_expense)
    flow = tips_inflation_flow_detail.copy()
    _require_columns(flow, ["quarter", "modeled_inflation_compensation_mil"])
    flow["modeled_inflation_compensation_mil"] = pd.to_numeric(
        flow["modeled_inflation_compensation_mil"],
        errors="coerce",
    )
    modeled = (
        flow.groupby("quarter", sort=False, dropna=False)["modeled_inflation_compensation_mil"]
        .sum(min_count=1)
        .rename("modeled_interest_mil")
        .reset_index()
    )
    merged = official.merge(modeled, on="quarter", how="outer")
    merged["gap_mil"] = pd.to_numeric(merged["modeled_interest_mil"], errors="coerce") - pd.to_numeric(
        merged["official_interest_mil"],
        errors="coerce",
    )
    merged["abs_gap_mil"] = merged["gap_mil"].abs()
    merged["gap_pct"] = merged["gap_mil"] / pd.to_numeric(merged["official_interest_mil"], errors="coerce") * 100.0
    merged["abs_gap_pct"] = merged["gap_pct"].abs()
    merged["source_coverage_pct"] = 100.0
    merged["model_version"] = TIPS_INFLATION_FLOW_MODEL_VERSION
    merged["canonical_status"] = "candidate_aggregate_tips_inflation_compensation_event_flow"
    return (
        merged.loc[:, TIPS_INFLATION_RECONCILIATION_COLUMNS]
        .sort_values("quarter", kind="stable")
        .reset_index(drop=True)
    )


def build_tips_coupon_accrual_flow_detail(
    tips_inflation_adjustment_stocks: pd.DataFrame,
    tips_auction_events: pd.DataFrame,
    tips_cpi_reference_path: pd.DataFrame,
    *,
    start_quarter: str | None = None,
    end_quarter: str | None = None,
) -> pd.DataFrame:
    """Build a source-only TIPS coupon accrued-liability flow by quarter."""

    if (
        tips_inflation_adjustment_stocks is None
        or tips_inflation_adjustment_stocks.empty
        or tips_auction_events is None
        or tips_auction_events.empty
        or tips_cpi_reference_path is None
        or tips_cpi_reference_path.empty
    ):
        return pd.DataFrame(columns=TIPS_COUPON_FLOW_DETAIL_COLUMNS)
    _require_columns(
        tips_inflation_adjustment_stocks,
        ["record_date", "cusip", "issued_amt_mil", "redeemed_amt_mil"],
    )
    _require_columns(
        tips_auction_events,
        [
            "cusip",
            "issue_date",
            "maturity_date",
            "total_accepted_mil",
            "coupon_rate_decimal",
            "ref_cpi_on_dated_date",
        ],
    )
    _require_columns(tips_cpi_reference_path, ["index_date", "ref_cpi"])
    stocks = tips_inflation_adjustment_stocks.copy()
    stocks["record_date"] = pd.to_datetime(stocks["record_date"], errors="coerce").dt.normalize()
    stocks["cusip"] = stocks["cusip"].astype(str).str.strip().str.upper()
    stocks["remaining_par_mil"] = (
        pd.to_numeric(stocks["issued_amt_mil"], errors="coerce")
        + pd.to_numeric(stocks["redeemed_amt_mil"], errors="coerce").fillna(0.0)
    )
    stock_by_cusip = (
        stocks.dropna(subset=["record_date", "cusip"])
        .groupby(["cusip", "record_date"], sort=False, dropna=False)["remaining_par_mil"]
        .sum(min_count=1)
        .reset_index()
        .sort_values(["cusip", "record_date"], kind="stable")
    )
    events = tips_auction_events.copy()
    events["cusip"] = events["cusip"].astype(str).str.strip().str.upper()
    events["issue_date"] = pd.to_datetime(events["issue_date"], errors="coerce").dt.normalize()
    events["maturity_date"] = pd.to_datetime(events["maturity_date"], errors="coerce").dt.normalize()
    for column in [
        "total_accepted_mil",
        "coupon_rate_decimal",
        "ref_cpi_on_dated_date",
        "adj_accrued_int_per1000",
    ]:
        if column in events.columns:
            events[column] = pd.to_numeric(events[column], errors="coerce")
    terms = (
        events.dropna(subset=["cusip"])
        .sort_values(["cusip", "issue_date"], kind="stable")
        .groupby("cusip", sort=False, dropna=False)
        .agg(
            coupon_rate_decimal=("coupon_rate_decimal", "first"),
            ref_cpi_on_dated_date=("ref_cpi_on_dated_date", "first"),
            maturity_date=("maturity_date", "max"),
        )
    )
    if terms.empty:
        return pd.DataFrame(columns=TIPS_COUPON_FLOW_DETAIL_COLUMNS)
    cpi = tips_cpi_reference_path.copy()
    cpi["index_date"] = pd.to_datetime(cpi["index_date"], errors="coerce").dt.normalize()
    cpi["ref_cpi"] = pd.to_numeric(cpi["ref_cpi"], errors="coerce")
    cpi_by_date = (
        cpi.dropna(subset=["index_date", "ref_cpi"])
        .drop_duplicates("index_date")
        .set_index("index_date")["ref_cpi"]
    )
    if cpi_by_date.empty:
        return pd.DataFrame(columns=TIPS_COUPON_FLOW_DETAIL_COLUMNS)
    issue_ai = pd.Series(dtype=float)
    if "adj_accrued_int_per1000" in events.columns:
        issue_work = events.copy()
        issue_work["quarter"] = issue_work["issue_date"].dt.to_period("Q").astype(str)
        issue_work["issue_accrued_interest_mil"] = (
            issue_work["total_accepted_mil"]
            * issue_work["adj_accrued_int_per1000"].fillna(0.0)
            / 1000.0
        )
        issue_ai = issue_work.groupby("quarter", sort=False)["issue_accrued_interest_mil"].sum(min_count=1)
    start_period = pd.Period(start_quarter, freq="Q") if start_quarter else cpi_by_date.index.min().to_period("Q")
    end_period = pd.Period(end_quarter, freq="Q") if end_quarter else cpi_by_date.index.max().to_period("Q")
    cpi_start_period = cpi_by_date.index.min().to_period("Q")
    if start_period < cpi_start_period:
        start_period = cpi_start_period
    rows: list[dict[str, object]] = []
    active_cusips = sorted(set(terms.index).intersection(set(stock_by_cusip["cusip"])))
    for period in pd.period_range(start_period, end_period, freq="Q"):
        q_start = period.start_time.normalize()
        q_open = q_start - pd.Timedelta(days=1)
        q_end = period.end_time.normalize()
        cash_coupon = 0.0
        accrued_open = 0.0
        accrued_close = 0.0
        for cusip in active_cusips:
            term = terms.loc[cusip]
            accrued_open += _tips_coupon_accrued_interest_at(
                cusip,
                q_open,
                term,
                stock_by_cusip,
                cpi_by_date,
            )
            accrued_close += _tips_coupon_accrued_interest_at(
                cusip,
                q_end,
                term,
                stock_by_cusip,
                cpi_by_date,
            )
            for coupon_date in _tips_coupon_dates_between(cusip, q_open, q_end, term):
                cash_coupon += _tips_coupon_cash_at(cusip, coupon_date, term, stock_by_cusip, cpi_by_date)
        issue_accrued = float(issue_ai.get(str(period), 0.0)) if not issue_ai.empty else 0.0
        modeled = cash_coupon + accrued_close - accrued_open - issue_accrued
        rows.append(
            {
                "quarter": str(period),
                "component": "tips_coupon_accrual",
                "cash_coupon_mil": cash_coupon,
                "accrued_interest_open_mil": accrued_open,
                "accrued_interest_close_mil": accrued_close,
                "issue_accrued_interest_mil": issue_accrued,
                "modeled_interest_mil": modeled,
                "source_coverage_status": "tips_coupon_cash_and_accrued_liability_from_mspd_auction_cpi",
                "model_version": TIPS_COUPON_FLOW_MODEL_VERSION,
                "canonical_status": "candidate_aggregate_tips_coupon_accrued_liability_timing_caveated",
            }
        )
    if not rows:
        return pd.DataFrame(columns=TIPS_COUPON_FLOW_DETAIL_COLUMNS)
    return pd.DataFrame(rows, columns=TIPS_COUPON_FLOW_DETAIL_COLUMNS)


def build_interest_component_reconciliation(
    treasury_interest_expense: pd.DataFrame,
    interest_detail: pd.DataFrame,
    *,
    bill_interest_flow_detail: pd.DataFrame | None = None,
    fixed_coupon_flow_detail: pd.DataFrame | None = None,
    frn_interest_flow_detail: pd.DataFrame | None = None,
    tips_inflation_flow_detail: pd.DataFrame | None = None,
    tips_coupon_flow_detail: pd.DataFrame | None = None,
    nonbill_discount_premium_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare official, stock-only, flow, and selected aggregate interest components."""

    official = _official_treasury_interest_pools(treasury_interest_expense).rename(
        columns={
            "official_pool": "component",
            "official_interest_expense_mil": "official_interest_mil",
        }
    )
    stock = _replay_interest_pools(interest_detail).rename(
        columns={
            "official_pool": "component",
            "replay_component_interest_mil": "stock_only_interest_mil",
        }
    )
    flow = pd.concat(
        [
            _bill_flow_interest_pool(bill_interest_flow_detail),
            _fixed_coupon_flow_interest_pool(fixed_coupon_flow_detail),
            _frn_flow_interest_pool(frn_interest_flow_detail),
            _tips_inflation_flow_interest_pool(tips_inflation_flow_detail),
            _tips_coupon_flow_interest_pool(tips_coupon_flow_detail),
            _nonbill_discount_premium_flow_interest_pool(nonbill_discount_premium_detail),
        ],
        ignore_index=True,
    ).rename(
        columns={
            "official_pool": "component",
            "replay_component_interest_mil": "flow_interest_mil",
        }
    )
    merged = official.merge(
        stock[["quarter", "component", "stock_only_interest_mil"]],
        on=["quarter", "component"],
        how="outer",
    )
    merged = merged.merge(
        flow[["quarter", "component", "flow_interest_mil"]],
        on=["quarter", "component"],
        how="outer",
    )
    if merged.empty:
        return pd.DataFrame(columns=INTEREST_COMPONENT_RECONCILIATION_COLUMNS)
    merged["flow_interest_mil"] = pd.to_numeric(merged["flow_interest_mil"], errors="coerce")
    merged["stock_only_interest_mil"] = pd.to_numeric(merged["stock_only_interest_mil"], errors="coerce")
    merged["scope_id"] = "marketable_coupon_like_ex_tips_inflation"
    merged.loc[
        merged["component"].isin(["tips_inflation_compensation", "nonbill_discount_premium"]),
        "scope_id",
    ] = "full_marketable_accrual"
    use_bill_flow = merged["component"].eq("bill_discount") & merged["flow_interest_mil"].notna()
    use_fixed_coupon_flow = merged["component"].eq("coupon_accrual") & merged["flow_interest_mil"].notna()
    use_frn_flow = merged["component"].eq("frn_interest") & merged["flow_interest_mil"].notna()
    use_tips_inflation_flow = (
        merged["component"].eq("tips_inflation_compensation") & merged["flow_interest_mil"].notna()
    )
    use_tips_coupon_flow = merged["component"].eq("tips_coupon_accrual") & merged["flow_interest_mil"].notna()
    use_nonbill_dp_flow = merged["component"].eq("nonbill_discount_premium") & merged["flow_interest_mil"].notna()
    use_flow = (
        use_bill_flow
        | use_fixed_coupon_flow
        | use_frn_flow
        | use_tips_inflation_flow
        | use_tips_coupon_flow
        | use_nonbill_dp_flow
    )
    merged["selected_model_interest_mil"] = merged["stock_only_interest_mil"]
    merged.loc[use_flow, "selected_model_interest_mil"] = merged.loc[use_flow, "flow_interest_mil"]
    merged["selected_model_basis"] = "stock_only_period_end"
    merged.loc[use_bill_flow, "selected_model_basis"] = "auction_lot_discount_flow"
    merged.loc[use_fixed_coupon_flow, "selected_model_basis"] = "mspd_auction_event_coupon_flow"
    merged.loc[use_frn_flow, "selected_model_basis"] = "frn_auction_lot_benchmark_accrual_flow"
    merged.loc[use_tips_inflation_flow, "selected_model_basis"] = "mspd_cusip_inflation_adjustment_event_flow"
    merged.loc[use_tips_coupon_flow, "selected_model_basis"] = "tips_coupon_accrued_liability_flow"
    merged.loc[use_nonbill_dp_flow, "selected_model_basis"] = "nonbill_effective_interest_discount_premium_flow"
    merged["gap_mil"] = (
        pd.to_numeric(merged["selected_model_interest_mil"], errors="coerce")
        - pd.to_numeric(merged["official_interest_mil"], errors="coerce")
    )
    merged["gap_pct"] = merged["gap_mil"] / pd.to_numeric(
        merged["official_interest_mil"],
        errors="coerce",
    ) * 100.0
    merged["source_coverage_pct"] = np.nan
    merged.loc[use_flow, "source_coverage_pct"] = 100.0
    merged["canonical_status"] = "stock_only_diagnostic_not_canonical"
    merged.loc[use_bill_flow, "canonical_status"] = "canonical_aggregate_bill_discount"
    merged.loc[use_fixed_coupon_flow, "canonical_status"] = "candidate_aggregate_fixed_coupon"
    merged.loc[use_frn_flow, "canonical_status"] = "candidate_aggregate_frn_interest"
    merged.loc[use_tips_inflation_flow, "canonical_status"] = (
        "candidate_aggregate_tips_inflation_compensation_event_flow"
    )
    merged.loc[use_tips_coupon_flow, "canonical_status"] = (
        "candidate_aggregate_tips_coupon_accrued_liability_timing_caveated"
    )
    merged.loc[use_nonbill_dp_flow, "canonical_status"] = "candidate_aggregate_nonbill_discount_premium"
    merged["exclusion_reason"] = pd.NA
    return (
        merged.loc[:, INTEREST_COMPONENT_RECONCILIATION_COLUMNS]
        .sort_values(["quarter", "component"], kind="stable")
        .reset_index(drop=True)
    )


def build_interest_component_certification(
    treasury_interest_expense: pd.DataFrame,
    interest_component_reconciliation: pd.DataFrame,
    nonbill_discount_premium_detail: pd.DataFrame | None = None,
    *,
    run_id: str = "historical_replay_interest_certification",
) -> pd.DataFrame:
    """Build fail-closed component certification rows for the narrowed quarterly scope."""

    official = _official_certification_components(treasury_interest_expense)
    model = _model_certification_components(
        interest_component_reconciliation,
        nonbill_discount_premium_detail,
    )
    merged = official.merge(model, on=["quarter", "component_id"], how="outer")
    if merged.empty:
        return pd.DataFrame(columns=INTEREST_COMPONENT_CERTIFICATION_COLUMNS)
    merged["calendar_year"] = merged["quarter"].astype(str).str.slice(0, 4).astype("Int64")
    merged["holdout_flag"] = merged["quarter"].astype(str).ge("2025Q1")
    merged["scope_id"] = CERTIFIED_QUARTERLY_SCOPE_ID
    merged["instrument_family"] = merged["instrument_family_x"].combine_first(merged["instrument_family_y"])
    merged["model_version"] = merged["model_version"].fillna("not_modeled")
    merged["official_interest_mil"] = pd.to_numeric(merged["official_interest_mil"], errors="coerce")
    merged["model_interest_mil"] = pd.to_numeric(merged["model_interest_mil"], errors="coerce")
    merged["gap_mil"] = merged["model_interest_mil"] - merged["official_interest_mil"]
    merged["abs_gap_mil"] = merged["gap_mil"].abs()
    merged["ape_pct"] = merged["abs_gap_mil"] / merged["official_interest_mil"].abs().replace(0.0, np.nan) * 100.0
    merged["source_coverage_pct"] = merged["source_coverage_pct"].fillna(0.0)
    included_components = {
        "bill_discount",
        "coupon_accrual",
        "frn_interest",
        "tips_inflation_compensation",
    }
    merged["included_in_scope"] = merged["component_id"].isin(included_components)
    merged["exclusion_reason_code"] = pd.NA
    merged.loc[merged["component_id"].eq("tips_coupon_accrual"), "exclusion_reason_code"] = "QUARTERLY_TIMING_CAVEATED"
    merged.loc[merged["component_id"].eq("frn_discount_premium"), "exclusion_reason_code"] = "SOURCE_IDENTITY_UNRESOLVED"
    merged.loc[
        merged["component_id"].isin(["nominal_nonbill_discount_premium", "tips_discount_premium"]),
        "exclusion_reason_code",
    ] = "PREMATURITY_REDEMPTION_ACCOUNTING_UNRESOLVED"
    merged["source_identity_status"] = merged["source_identity_status"].fillna("not_modeled")
    merged["source_price_tier"] = merged["source_price_tier"].fillna(pd.NA)
    merged["lot_count"] = pd.to_numeric(merged["lot_count"], errors="coerce").fillna(0).astype(int)
    merged["conservation_residual_mil"] = pd.to_numeric(merged["conservation_residual_mil"], errors="coerce")
    merged["terminal_carrying_residual_max_per100"] = pd.to_numeric(
        merged["terminal_carrying_residual_max_per100"],
        errors="coerce",
    )
    merged["fixed_control_residual_mil"] = np.nan
    merged["timing_status"] = "quarterly"
    merged.loc[merged["component_id"].eq("tips_coupon_accrual"), "timing_status"] = "quarterly_timing_caveated"
    merged.loc[
        merged["component_id"].isin(
            [
                "frn_discount_premium",
                "nominal_nonbill_discount_premium",
                "tips_discount_premium",
            ]
        ),
        "timing_status",
    ] = "not_in_selected_scope"
    thresholds = {
        "bill_discount": (2.0, 200.0),
        "coupon_accrual": (0.1, 75.0),
        "frn_interest": (0.1, 5.0),
        "tips_inflation_compensation": (2.0, 300.0),
        "nominal_nonbill_discount_premium": (3.5, 50.0),
        "tips_discount_premium": (1.0, 15.0),
    }
    merged["quarter_gate_status"] = "not_applicable"
    for component_id, (ape_limit, abs_limit) in thresholds.items():
        mask = merged["component_id"].eq(component_id)
        passes = (
            merged["model_interest_mil"].notna()
            & merged["official_interest_mil"].notna()
            & merged["ape_pct"].le(ape_limit)
            & merged["abs_gap_mil"].le(abs_limit)
        )
        merged.loc[mask, "quarter_gate_status"] = np.where(
            passes.loc[mask],
            "pass",
            "fail",
        )
    merged["annual_gate_status"] = merged["quarter_gate_status"]
    merged["holdout_gate_status"] = np.where(merged["holdout_flag"], merged["quarter_gate_status"], "not_holdout")
    merged["certification_status"] = "failed"
    included_pass = merged["included_in_scope"] & merged["quarter_gate_status"].eq("pass")
    merged.loc[included_pass, "certification_status"] = "certified_quarterly"
    merged.loc[merged["component_id"].eq("tips_coupon_accrual"), "certification_status"] = "candidate_timing_caveated"
    merged.loc[
        merged["component_id"].isin(
            [
                "frn_discount_premium",
                "nominal_nonbill_discount_premium",
                "tips_discount_premium",
            ]
        ),
        "certification_status",
    ] = "excluded_selected_scope"
    merged["failure_code"] = pd.NA
    merged.loc[merged["included_in_scope"] & merged["model_interest_mil"].isna(), "failure_code"] = "MISSING_MODEL_COMPONENT"
    merged.loc[merged["included_in_scope"] & merged["quarter_gate_status"].eq("fail"), "failure_code"] = "QUARTER_GATE_FAIL"
    merged.loc[merged["component_id"].eq("tips_coupon_accrual"), "failure_code"] = "QUARTERLY_TIMING_CAVEATED"
    merged.loc[merged["component_id"].eq("frn_discount_premium"), "failure_code"] = "SOURCE_IDENTITY_UNRESOLVED"
    merged.loc[
        merged["component_id"].isin(["nominal_nonbill_discount_premium", "tips_discount_premium"]),
        "failure_code",
    ] = "PREMATURITY_REDEMPTION_ACCOUNTING_UNRESOLVED"
    merged["schema_version"] = INTEREST_COMPONENT_CERTIFICATION_SCHEMA_VERSION
    merged["run_id"] = run_id
    merged["evidence_artifact"] = merged["evidence_artifact"].fillna("historical_replay_interest_component_reconciliation.csv")
    input_hash = _frame_sha256(treasury_interest_expense)
    output_hash = _frame_sha256(merged[[column for column in merged.columns if column != "output_sha256"]])
    merged["input_sha256"] = input_hash
    merged["output_sha256"] = output_hash
    return (
        merged.loc[:, INTEREST_COMPONENT_CERTIFICATION_COLUMNS]
        .sort_values(["quarter", "component_id"], kind="stable")
        .reset_index(drop=True)
    )


def build_interest_scope_certification(
    component_certification: pd.DataFrame,
    *,
    run_id: str = "historical_replay_interest_certification",
) -> pd.DataFrame:
    """Build a narrowed quarterly scope certification from component certification rows."""

    if component_certification is None or component_certification.empty:
        return pd.DataFrame(columns=INTEREST_SCOPE_CERTIFICATION_COLUMNS)
    _require_columns(
        component_certification,
        [
            "quarter",
            "component_id",
            "official_interest_mil",
            "model_interest_mil",
            "certification_status",
            "included_in_scope",
        ],
    )
    expected_components = [
        "bill_discount",
        "coupon_accrual",
        "frn_interest",
        "tips_inflation_compensation",
    ]
    rows = []
    component_hash = _frame_sha256(component_certification)
    for quarter, group in component_certification.groupby("quarter", sort=True, dropna=False):
        present = sorted(group["component_id"].astype(str).unique().tolist())
        certified = sorted(
            group.loc[
                group["included_in_scope"].astype(bool)
                & group["certification_status"].astype(str).eq("certified_quarterly"),
                "component_id",
            ].astype(str).unique().tolist()
        )
        excluded = sorted(
            group.loc[~group["included_in_scope"].astype(bool), "component_id"].astype(str).unique().tolist()
        )
        included = group.loc[group["included_in_scope"].astype(bool)].copy()
        official_scope_total = pd.to_numeric(included["official_interest_mil"], errors="coerce").sum(min_count=1)
        model_scope_total = pd.to_numeric(included["model_interest_mil"], errors="coerce").sum(min_count=1)
        excluded_official = pd.to_numeric(
            group.loc[~group["included_in_scope"].astype(bool), "official_interest_mil"],
            errors="coerce",
        ).sum(min_count=1)
        gap = model_scope_total - official_scope_total
        ape = abs(gap) / abs(official_scope_total) * 100.0 if pd.notna(official_scope_total) and official_scope_total != 0 else np.nan
        missing = sorted(set(expected_components) - set(present))
        all_certified = set(expected_components).issubset(set(certified))
        aggregate_pass = pd.notna(ape) and ape <= 0.5 and abs(gap) <= 1500.0
        rows.append(
            {
                "schema_version": INTEREST_SCOPE_CERTIFICATION_SCHEMA_VERSION,
                "run_id": run_id,
                "quarter": str(quarter),
                "scope_id": CERTIFIED_QUARTERLY_SCOPE_ID,
                "scope_claim": "quarterly certified core marketable interest excluding quarterly TIPS coupon and nonbill discount/premium",
                "expected_components": ";".join(expected_components),
                "present_components": ";".join(present),
                "certified_components": ";".join(certified),
                "excluded_components": ";".join(excluded),
                "official_scope_total_mil": official_scope_total,
                "model_scope_total_mil": model_scope_total,
                "official_excluded_amount_mil": excluded_official,
                "gap_mil": gap,
                "ape_pct": ape,
                "completeness_status": "complete" if not missing else "missing:" + ";".join(missing),
                "aggregate_gate_status": "pass" if aggregate_pass else "fail",
                "fail_closed_status": "pass" if all_certified and aggregate_pass else "fail_closed",
                "certification_status": "certified_quarterly" if all_certified and aggregate_pass else "failed",
                "failure_code": pd.NA if all_certified and aggregate_pass else "COMPONENT_OR_AGGREGATE_GATE_FAIL",
                "component_manifest_sha256": component_hash,
            }
        )
    return pd.DataFrame(rows, columns=INTEREST_SCOPE_CERTIFICATION_COLUMNS)


def build_official_interest_scope_bridge(treasury_interest_expense: pd.DataFrame) -> pd.DataFrame:
    """Map every official Treasury interest row to replay interest scopes."""

    if treasury_interest_expense is None or treasury_interest_expense.empty:
        return pd.DataFrame(columns=OFFICIAL_INTEREST_SCOPE_BRIDGE_COLUMNS)
    required = {"record_date", "expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}
    if not required.issubset(treasury_interest_expense.columns):
        return pd.DataFrame(columns=OFFICIAL_INTEREST_SCOPE_BRIDGE_COLUMNS)
    working = treasury_interest_expense.copy()
    working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    classifications = working.apply(_classify_official_interest_scope, axis=1, result_type="expand")
    working = pd.concat([working, classifications], axis=1)
    return (
        working.loc[:, OFFICIAL_INTEREST_SCOPE_BRIDGE_COLUMNS]
        .sort_values(
            ["quarter", "expense_catg_desc", "expense_group_desc", "expense_type_desc", "record_date"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _iter_snapshots(
    portfolio_snapshots: Mapping[str, pd.DataFrame] | pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    if isinstance(portfolio_snapshots, pd.DataFrame):
        _require_columns(portfolio_snapshots, ["quarter"])
        snapshots: list[tuple[str, pd.DataFrame]] = []
        for quarter, frame in portfolio_snapshots.groupby("quarter", sort=True, dropna=False):
            snapshots.append((str(quarter), frame.copy()))
        return snapshots
    snapshots = []
    for quarter, frame in portfolio_snapshots.items():
        working = frame.copy()
        if "quarter" not in working.columns:
            working["quarter"] = str(quarter)
        snapshots.append((str(quarter), working))
    snapshots.sort(key=lambda item: pd.Period(str(item[0]), freq="Q"))
    return snapshots


def _official_treasury_interest_pools(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "official_interest_expense_mil", "official_row_count"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "quarter" not in working.columns:
        if "record_date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    public_issue = working["expense_catg_desc"].astype(str).str.upper().eq("INTEREST EXPENSE ON PUBLIC ISSUES")
    working = working.loc[public_issue].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["official_pool"] = working.apply(_official_interest_pool, axis=1)
    working = working.loc[working["official_pool"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    out = (
        working.groupby(["quarter", "official_pool"], dropna=False, sort=False)
        .agg(
            official_interest_expense_mil=("month_expense_amt", lambda values: values.sum(min_count=1) / 1_000_000.0),
            official_row_count=("month_expense_amt", "size"),
        )
        .reset_index()
    )
    return out.loc[:, columns]


def _official_interest_pool(row: pd.Series):
    group = str(row.get("expense_group_desc", "")).strip().lower()
    expense_type = str(row.get("expense_type_desc", "")).strip().lower()
    if "int. expense inflation compensation" in expense_type:
        return "tips_inflation_compensation"
    if "treasury bills" in expense_type and group == "amortized discount":
        return "bill_discount"
    if "treasury floating rate notes" in expense_type and group == "accrued interest expense":
        return "frn_interest"
    if "inflation protected securities" in expense_type and group == "accrued interest expense":
        return "tips_coupon_accrual"
    if expense_type in {"treasury notes", "treasury bonds"} and group == "accrued interest expense":
        return "coupon_accrual"
    if group in {"amortized discount", "amortized premium"} and expense_type in {
        "treasury notes",
        "treasury bonds",
        "treasury floating rate notes (frn)",
        "treasury inflation protected securities (tips)",
    }:
        return "nonbill_discount_premium"
    return pd.NA


def _official_certification_components(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "component_id", "instrument_family", "official_interest_mil"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    if not {"expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "quarter" not in working.columns:
        if "record_date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    public_issue = working["expense_catg_desc"].astype(str).str.upper().eq("INTEREST EXPENSE ON PUBLIC ISSUES")
    working = working.loc[public_issue].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    group = working["expense_group_desc"].astype(str).str.strip().str.lower()
    expense_type = working["expense_type_desc"].astype(str).str.strip().str.lower()
    working["component_id"] = pd.NA
    working["instrument_family"] = pd.NA
    working.loc[expense_type.eq("treasury bills") & group.eq("amortized discount"), "component_id"] = "bill_discount"
    working.loc[working["component_id"].eq("bill_discount"), "instrument_family"] = "Treasury Bills"
    working.loc[expense_type.isin(["treasury notes", "treasury bonds"]) & group.eq("accrued interest expense"), "component_id"] = "coupon_accrual"
    working.loc[working["component_id"].eq("coupon_accrual"), "instrument_family"] = "Treasury Notes/Bonds"
    working.loc[expense_type.str.contains("treasury floating rate notes", na=False) & group.eq("accrued interest expense"), "component_id"] = "frn_interest"
    working.loc[working["component_id"].eq("frn_interest"), "instrument_family"] = "Treasury Floating Rate Notes (FRN)"
    working.loc[expense_type.eq("int. expense inflation compensation (tips)") & group.eq("accrued interest expense"), "component_id"] = "tips_inflation_compensation"
    working.loc[working["component_id"].eq("tips_inflation_compensation"), "instrument_family"] = "Treasury Inflation Protected Securities (TIPS)"
    tips_coupon = expense_type.str.contains("inflation protected securities", na=False) & group.eq("accrued interest expense")
    tips_coupon &= ~expense_type.eq("int. expense inflation compensation (tips)")
    working.loc[tips_coupon, "component_id"] = "tips_coupon_accrual"
    working.loc[working["component_id"].eq("tips_coupon_accrual"), "instrument_family"] = "Treasury Inflation Protected Securities (TIPS)"
    nonbill_group = group.isin(["amortized discount", "amortized premium"])
    working.loc[nonbill_group & expense_type.isin(["treasury notes", "treasury bonds"]), "component_id"] = "nominal_nonbill_discount_premium"
    working.loc[working["component_id"].eq("nominal_nonbill_discount_premium"), "instrument_family"] = "Treasury Notes/Bonds"
    working.loc[nonbill_group & expense_type.eq("treasury inflation protected securities (tips)"), "component_id"] = "tips_discount_premium"
    working.loc[working["component_id"].eq("tips_discount_premium"), "instrument_family"] = "Treasury Inflation Protected Securities (TIPS)"
    working.loc[nonbill_group & expense_type.eq("treasury floating rate notes (frn)"), "component_id"] = "frn_discount_premium"
    working.loc[working["component_id"].eq("frn_discount_premium"), "instrument_family"] = "Treasury Floating Rate Notes (FRN)"
    working = working.loc[working["component_id"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    return (
        working.groupby(["quarter", "component_id", "instrument_family"], dropna=False, sort=False)["month_expense_amt"]
        .sum(min_count=1)
        .div(1_000_000.0)
        .rename("official_interest_mil")
        .reset_index()
        .loc[:, columns]
    )


def _model_certification_components(
    interest_component_reconciliation: pd.DataFrame | None,
    nonbill_discount_premium_detail: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "quarter",
        "component_id",
        "instrument_family",
        "model_interest_mil",
        "source_coverage_pct",
        "source_identity_status",
        "source_price_tier",
        "lot_count",
        "conservation_residual_mil",
        "terminal_carrying_residual_max_per100",
        "model_version",
        "evidence_artifact",
    ]
    rows = []
    if interest_component_reconciliation is not None and not interest_component_reconciliation.empty:
        base_map = {
            "bill_discount": ("Treasury Bills", "historical_replay_interest_component_reconciliation.csv"),
            "coupon_accrual": ("Treasury Notes/Bonds", "historical_replay_interest_component_reconciliation.csv"),
            "frn_interest": ("Treasury Floating Rate Notes (FRN)", "historical_replay_interest_component_reconciliation.csv"),
            "tips_inflation_compensation": ("Treasury Inflation Protected Securities (TIPS)", "historical_replay_interest_component_reconciliation.csv"),
            "tips_coupon_accrual": ("Treasury Inflation Protected Securities (TIPS)", "historical_replay_interest_component_reconciliation.csv"),
        }
        base = interest_component_reconciliation.copy()
        for component_id, (family, artifact) in base_map.items():
            subset = base.loc[base["component"].astype(str).eq(component_id)].copy()
            for _, row in subset.iterrows():
                rows.append(
                    {
                        "quarter": row.get("quarter"),
                        "component_id": component_id,
                        "instrument_family": family,
                        "model_interest_mil": row.get("selected_model_interest_mil"),
                        "source_coverage_pct": row.get("source_coverage_pct"),
                        "source_identity_status": row.get("canonical_status"),
                        "source_price_tier": pd.NA,
                        "lot_count": np.nan,
                        "conservation_residual_mil": np.nan,
                        "terminal_carrying_residual_max_per100": np.nan,
                        "model_version": row.get("selected_model_basis"),
                        "evidence_artifact": artifact,
                    }
                )
    if nonbill_discount_premium_detail is not None and not nonbill_discount_premium_detail.empty:
        detail = nonbill_discount_premium_detail.copy()
        detail["quarter_amortization_mil"] = pd.to_numeric(detail["quarter_amortization_mil"], errors="coerce")
        detail["terminal_residual_per100"] = pd.to_numeric(detail["terminal_residual_per100"], errors="coerce")
        detail["component_id"] = np.where(
            detail["instrument_family"].astype(str).eq("Treasury Inflation Protected Securities (TIPS)"),
            "tips_discount_premium",
            "nominal_nonbill_discount_premium",
        )
        grouped = (
            detail.groupby(["quarter", "component_id"], dropna=False, sort=False)
            .agg(
                instrument_family=("instrument_family", lambda values: "Treasury Notes/Bonds" if values.astype(str).isin(["Treasury Notes", "Treasury Bonds"]).any() else values.dropna().astype(str).iloc[0]),
                model_interest_mil=("quarter_amortization_mil", lambda values: values.sum(min_count=1)),
                source_coverage_pct=("quarter_amortization_mil", lambda values: 100.0),
                source_identity_status=("source_identity_status", _first_non_null_value),
                source_price_tier=("price_source_tier", lambda values: ";".join(sorted(values.dropna().astype(str).unique().tolist()))),
                lot_count=("lot_id", pd.Series.nunique),
                conservation_residual_mil=("quarter_amortization_mil", lambda values: 0.0),
                terminal_carrying_residual_max_per100=("terminal_residual_per100", lambda values: values.abs().max()),
                model_version=("model_version", _first_non_null_value),
            )
            .reset_index()
        )
        grouped["evidence_artifact"] = "historical_replay_nonbill_discount_premium_lot_detail.csv"
        rows.extend(grouped.to_dict("records"))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).loc[:, columns]


def _classify_official_interest_scope(row: pd.Series) -> pd.Series:
    category = str(row.get("expense_catg_desc", "")).strip().lower()
    group = str(row.get("expense_group_desc", "")).strip().lower()
    expense_type = str(row.get("expense_type_desc", "")).strip().lower()
    scope_id = "excluded"
    component = pd.NA
    coupon_like = False
    full_marketable = False
    exclusion_reason = pd.NA

    if category == "interest expense on govt account series":
        scope_id = "government_account_series"
        component = "gas_interest_expense"
        exclusion_reason = "government_account_series_outside_marketable_replay"
    elif category != "interest expense on public issues":
        exclusion_reason = "unknown_interest_expense_category"
    elif group == "accrued interest expense":
        if expense_type in {"treasury notes", "treasury bonds"}:
            scope_id = "marketable_coupon_like_ex_tips_inflation"
            component = "coupon_accrual"
            coupon_like = True
            full_marketable = True
        elif "treasury floating rate notes" in expense_type:
            scope_id = "marketable_coupon_like_ex_tips_inflation"
            component = "frn_interest"
            coupon_like = True
            full_marketable = True
        elif "inflation protected securities" in expense_type and "inflation compensation" not in expense_type:
            scope_id = "marketable_coupon_like_ex_tips_inflation"
            component = "tips_coupon_accrual"
            coupon_like = True
            full_marketable = True
        elif "int. expense inflation compensation" in expense_type:
            scope_id = "full_marketable_accrual"
            component = "tips_inflation_compensation"
            full_marketable = True
            exclusion_reason = "excluded_from_coupon_like_scope_not_full_marketable_scope"
        else:
            scope_id = "excluded_public_nonmarketable_or_misc"
            component = "public_nonmarketable_accrued_interest"
            exclusion_reason = "public_issue_nonmarketable_or_misc_outside_marketable_replay"
    elif group == "amortized discount":
        if expense_type == "treasury bills":
            scope_id = "marketable_coupon_like_ex_tips_inflation"
            component = "bill_discount"
            coupon_like = True
            full_marketable = True
        elif expense_type in {
            "treasury notes",
            "treasury bonds",
            "treasury floating rate notes (frn)",
            "treasury inflation protected securities (tips)",
        }:
            scope_id = "full_marketable_accrual"
            component = "nonbill_discount_premium"
            full_marketable = True
            exclusion_reason = "excluded_from_coupon_like_scope_not_full_marketable_scope"
        else:
            scope_id = "excluded_public_nonmarketable_or_misc"
            component = "public_nonmarketable_discount"
            exclusion_reason = "public_issue_nonmarketable_or_misc_outside_marketable_replay"
    elif group == "amortized premium":
        if expense_type in {
            "treasury notes",
            "treasury bonds",
            "treasury floating rate notes (frn)",
            "treasury inflation protected securities (tips)",
        }:
            scope_id = "full_marketable_accrual"
            component = "nonbill_discount_premium"
            full_marketable = True
            exclusion_reason = "excluded_from_coupon_like_scope_not_full_marketable_scope"
        else:
            scope_id = "excluded_public_nonmarketable_or_misc"
            component = "public_nonmarketable_premium"
            exclusion_reason = "public_issue_nonmarketable_or_misc_outside_marketable_replay"
    elif group == "savings bonds":
        scope_id = "excluded_public_nonmarketable_or_misc"
        component = "savings_bonds"
        exclusion_reason = "savings_bonds_outside_marketable_replay"
    elif group == "miscellaneous interest expense":
        scope_id = "excluded_public_nonmarketable_or_misc"
        component = "miscellaneous_public_interest"
        exclusion_reason = "miscellaneous_public_interest_outside_marketable_replay"
    else:
        scope_id = "unmapped"
        exclusion_reason = "unmapped_official_interest_row"

    return pd.Series(
        {
            "scope_id": scope_id,
            "component": component,
            "include_in_coupon_like_scope": bool(coupon_like),
            "include_in_full_marketable_accrual_scope": bool(full_marketable),
            "exclusion_reason": exclusion_reason,
        }
    )


def _replay_interest_pools(detail: pd.DataFrame) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty or not {"quarter", "component", "interest_amount"}.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    if "excluded_from_default_canonical" in working.columns:
        working = working.loc[~working["excluded_from_default_canonical"].fillna(False)].copy()
    component_map = {
        "bill_discount_amortization": "bill_discount",
        "fixed_coupon_accrual": "coupon_accrual",
        "frn_accrued_interest_pass_through": "frn_interest",
        "tips_coupon_accrual": "tips_coupon_accrual",
    }
    working["official_pool"] = working["component"].map(component_map)
    working = working.loc[working["official_pool"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["interest_amount"] = pd.to_numeric(working["interest_amount"], errors="coerce")
    return (
        working.groupby(["quarter", "official_pool"], dropna=False, sort=False)["interest_amount"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
        .loc[:, columns]
    )


def _bill_flow_interest_pool(detail: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "component", "modeled_interest_mil"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.loc[detail["component"].astype(str).eq("bill_discount")].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["modeled_interest_mil"] = pd.to_numeric(working["modeled_interest_mil"], errors="coerce")
    out = (
        working.groupby("quarter", dropna=False, sort=False)["modeled_interest_mil"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
    )
    out["official_pool"] = "bill_discount"
    return out.loc[:, columns]


def _fixed_coupon_flow_interest_pool(detail: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "modeled_interest_mil"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    working["modeled_interest_mil"] = pd.to_numeric(working["modeled_interest_mil"], errors="coerce")
    out = (
        working.groupby("quarter", dropna=False, sort=False)["modeled_interest_mil"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
    )
    out["official_pool"] = "coupon_accrual"
    return out.loc[:, columns]


def _frn_flow_interest_pool(detail: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "modeled_interest_mil"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    working["modeled_interest_mil"] = pd.to_numeric(working["modeled_interest_mil"], errors="coerce")
    out = (
        working.groupby("quarter", dropna=False, sort=False)["modeled_interest_mil"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
    )
    out["official_pool"] = "frn_interest"
    return out.loc[:, columns]


def _tips_inflation_flow_interest_pool(detail: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "modeled_inflation_compensation_mil"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    working["modeled_inflation_compensation_mil"] = pd.to_numeric(
        working["modeled_inflation_compensation_mil"],
        errors="coerce",
    )
    out = (
        working.groupby("quarter", dropna=False, sort=False)["modeled_inflation_compensation_mil"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
    )
    out["official_pool"] = "tips_inflation_compensation"
    return out.loc[:, columns]


def _tips_coupon_flow_interest_pool(detail: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "modeled_interest_mil"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    working["modeled_interest_mil"] = pd.to_numeric(working["modeled_interest_mil"], errors="coerce")
    out = (
        working.groupby("quarter", dropna=False, sort=False)["modeled_interest_mil"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
    )
    out["official_pool"] = "tips_coupon_accrual"
    return out.loc[:, columns]


def _nonbill_discount_premium_flow_interest_pool(detail: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "replay_component_interest_mil"]
    if detail is None or detail.empty:
        return pd.DataFrame(columns=columns)
    required = {"quarter", "quarter_amortization_mil"}
    if not required.issubset(detail.columns):
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    working["quarter_amortization_mil"] = pd.to_numeric(
        working["quarter_amortization_mil"],
        errors="coerce",
    )
    out = (
        working.groupby("quarter", dropna=False, sort=False)["quarter_amortization_mil"]
        .sum(min_count=1)
        .rename("replay_component_interest_mil")
        .reset_index()
    )
    out["official_pool"] = "nonbill_discount_premium"
    return out.loc[:, columns]


def _official_fixed_coupon_by_class(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "security_class", "official_interest_mil"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "quarter" not in working.columns:
        if "record_date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    public_issue = working["expense_catg_desc"].astype(str).str.upper().eq("INTEREST EXPENSE ON PUBLIC ISSUES")
    accrued = working["expense_group_desc"].astype(str).str.upper().eq("ACCRUED INTEREST EXPENSE")
    type_text = working["expense_type_desc"].astype(str).str.strip()
    working = working.loc[public_issue & accrued & type_text.isin(["Treasury Notes", "Treasury Bonds"])].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["security_class"] = type_text.loc[working.index].map(
        {"Treasury Notes": "Notes", "Treasury Bonds": "Bonds"}
    )
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    by_class = (
        working.groupby(["quarter", "security_class"], dropna=False, sort=False)["month_expense_amt"]
        .sum(min_count=1)
        .div(1_000_000.0)
        .rename("official_interest_mil")
        .reset_index()
    )
    combined = (
        by_class.groupby("quarter", dropna=False, sort=False)["official_interest_mil"]
        .sum(min_count=1)
        .reset_index()
    )
    combined["security_class"] = "Notes+Bonds"
    return pd.concat([by_class, combined], ignore_index=True).loc[:, columns]


def _official_frn_interest_by_quarter(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_interest_mil"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "quarter" not in working.columns:
        if "record_date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    public_issue = working["expense_catg_desc"].astype(str).str.upper().eq("INTEREST EXPENSE ON PUBLIC ISSUES")
    accrued = working["expense_group_desc"].astype(str).str.upper().eq("ACCRUED INTEREST EXPENSE")
    frn_type = working["expense_type_desc"].astype(str).str.lower().str.contains("treasury floating rate notes", na=False)
    working = working.loc[public_issue & accrued & frn_type].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    return (
        working.groupby("quarter", dropna=False, sort=False)["month_expense_amt"]
        .sum(min_count=1)
        .div(1_000_000.0)
        .rename("official_interest_mil")
        .reset_index()
        .loc[:, columns]
    )


def _official_tips_inflation_by_quarter(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_interest_mil"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"expense_catg_desc", "expense_group_desc", "expense_type_desc", "month_expense_amt"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    working = frame.copy()
    if "quarter" not in working.columns:
        if "record_date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["record_date"], errors="coerce").dt.to_period("Q").astype(str)
    public_issue = working["expense_catg_desc"].astype(str).str.upper().eq("INTEREST EXPENSE ON PUBLIC ISSUES")
    accrued = working["expense_group_desc"].astype(str).str.upper().eq("ACCRUED INTEREST EXPENSE")
    tips_inflation = working["expense_type_desc"].astype(str).str.lower().eq(
        "int. expense inflation compensation (tips)"
    )
    working = working.loc[public_issue & accrued & tips_inflation].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["month_expense_amt"] = pd.to_numeric(working["month_expense_amt"], errors="coerce")
    return (
        working.groupby("quarter", dropna=False, sort=False)["month_expense_amt"]
        .sum(min_count=1)
        .div(1_000_000.0)
        .rename("official_interest_mil")
        .reset_index()
        .loc[:, columns]
    )


def _tdcest_candidate_interest_pools(candidate: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["quarter", "official_pool", "tdcest_candidate_component_mil"]
    if candidate is None or candidate.empty:
        return pd.DataFrame(columns=columns)
    required = {"component_family", "component_anchored_interest_mil"}
    if not required.issubset(candidate.columns):
        return pd.DataFrame(columns=columns)
    working = candidate.copy()
    if "quarter" not in working.columns:
        if "date" not in working.columns:
            return pd.DataFrame(columns=columns)
        working["quarter"] = pd.to_datetime(working["date"], errors="coerce").dt.to_period("Q").astype(str)
    family_map = {
        "bill_discount": "bill_discount",
        "coupon_accrual": "coupon_accrual",
        "frn_interest": "frn_interest",
    }
    working["official_pool"] = working["component_family"].map(family_map)
    working = working.loc[working["official_pool"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["component_anchored_interest_mil"] = pd.to_numeric(
        working["component_anchored_interest_mil"],
        errors="coerce",
    )
    return (
        working.groupby(["quarter", "official_pool"], dropna=False, sort=False)[
            "component_anchored_interest_mil"
        ]
        .sum(min_count=1)
        .rename("tdcest_candidate_component_mil")
        .reset_index()
        .loc[:, columns]
    )


def _build_quarter_interest_detail_frame(quarter: str, snapshot: pd.DataFrame) -> pd.DataFrame:
    if snapshot.empty:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    quarter_start, quarter_end, quarter_days = _quarter_bounds(quarter)
    frame = snapshot.copy()
    security_type = _text_from_columns(frame, "SecurityType", "security_type")
    active = security_type.str.strip().ne("")
    if not active.any():
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    frame = frame.loc[active].copy()
    security_type = security_type.loc[active]

    issue_date = pd.to_datetime(_value_from_columns(frame, "IssueDate", "issue_date"), errors="coerce").dt.normalize()
    maturity_date = pd.to_datetime(_value_from_columns(frame, "MaturityDate", "maturity_date"), errors="coerce").dt.normalize()
    exposure_start = pd.Series(quarter_start, index=frame.index, dtype="datetime64[ns]")
    issue_after_start = issue_date.notna() & (issue_date > quarter_start)
    exposure_start.loc[issue_after_start] = issue_date.loc[issue_after_start]
    exposure_end = pd.Series(quarter_end, index=frame.index, dtype="datetime64[ns]")
    maturity_before_end = maturity_date.notna() & (maturity_date < quarter_end)
    exposure_end.loc[maturity_before_end] = maturity_date.loc[maturity_before_end]
    days_held = ((exposure_end - exposure_start).dt.days + 1).where(exposure_end >= exposure_start, 0)
    held = days_held > 0
    if not held.any():
        return pd.DataFrame(columns=DETAIL_COLUMNS)

    frame = frame.loc[held].copy()
    security_type = security_type.loc[held]
    issue_date = issue_date.loc[held]
    maturity_date = maturity_date.loc[held]
    exposure_start = exposure_start.loc[held]
    exposure_end = exposure_end.loc[held]
    days_held = days_held.loc[held]

    face_value = _numeric_from_columns(frame, "FaceValue", "face_value", default=0.0)
    coupon_rate = _numeric_from_columns(frame, "CouponRate", "coupon_rate", default=0.0)
    issue_price_ratio = _numeric_from_columns(frame, "IssuePriceRatio", "issue_price_ratio", default=np.nan)
    issue_proceeds = _numeric_from_columns(frame, "IssueProceeds", "issue_proceeds", default=np.nan)
    inferred_price = issue_proceeds / face_value.replace(0.0, np.nan)
    issue_price_ratio = issue_price_ratio.where(issue_price_ratio.notna(), inferred_price)
    issue_yield = _numeric_from_columns(
        frame,
        "IssueYieldAtIssue",
        "issue_yield_decimal",
        "IssueYield",
        default=np.nan,
    )
    fixed_spread = _numeric_from_columns(frame, "FixedSpread", "fixed_spread", default=np.nan)
    index_ratio_end = _numeric_from_columns(
        frame,
        "IndexRatioEnd",
        "IndexRatio",
        "index_ratio_end",
        "index_ratio",
        default=np.nan,
    )
    index_ratio_start = _numeric_from_columns(frame, "IndexRatioStart", "index_ratio_start", default=np.nan)
    index_ratio_start = index_ratio_start.where(index_ratio_start.notna(), index_ratio_end)
    ref_cpi_start = _numeric_from_columns(frame, "ReferenceCPI_Start", "ref_cpi_start", default=np.nan)
    ref_cpi_end = _numeric_from_columns(frame, "ReferenceCPI_End", "ref_cpi_end", default=np.nan)
    base_cpi = _numeric_from_columns(frame, "ReferenceCPI_Issue", "reference_cpi_issue", default=np.nan)
    ref_cpi_start = ref_cpi_start.where(ref_cpi_start.notna(), ref_cpi_end)
    ref_cpi_end = ref_cpi_end.where(ref_cpi_end.notna(), ref_cpi_start)
    ref_cpi_start = ref_cpi_start.where(ref_cpi_start.notna(), base_cpi * index_ratio_start)
    ref_cpi_end = ref_cpi_end.where(ref_cpi_end.notna(), base_cpi * index_ratio_end)
    adjusted_principal = _numeric_from_columns(frame, "AdjustedPrincipal", "adjusted_principal", default=np.nan)
    original_principal = _numeric_from_columns(frame, "OriginalPrincipal", "original_principal", default=np.nan)
    derived_adjusted = original_principal * index_ratio_end
    adjusted_principal = adjusted_principal.where(adjusted_principal > TGA_FLOOR_TOLERANCE, derived_adjusted)
    adjusted_principal = adjusted_principal.where(adjusted_principal > TGA_FLOOR_TOLERANCE, face_value)

    broad_holder = _text_from_columns(frame, "broad_holder_class", "HolderType", "holder_type")
    holder_subbucket = _text_from_columns(frame, "HolderSubBucket", "holder_subbucket")
    tdcsim_holder = _text_from_columns(frame, "tdcsim_holder")
    private_with_subbucket = tdcsim_holder.eq("") & broad_holder.eq(_PRIVATE_PREFIX) & holder_subbucket.ne("")
    tdcsim_holder = tdcsim_holder.where(tdcsim_holder.ne(""), broad_holder)
    tdcsim_holder = tdcsim_holder.where(~private_with_subbucket, _PRIVATE_PREFIX + ":" + holder_subbucket)
    residual_holder = tdcsim_holder.astype(str).str.lower().str.replace(r"[^a-z0-9]+", "", regex=True).isin(
        {"sourcebasisresidual", "mspdz1sourcebasisresidual"}
    )
    face_begin = face_value.where(~(issue_date.notna() & (issue_date > quarter_start)), 0.0)
    face_end = face_value.where(~(maturity_date.notna() & (maturity_date < quarter_end)), 0.0)

    base = pd.DataFrame(
        {
            "quarter": quarter,
            "cusip/cohort_id": _first_nonblank_text(frame, "cohort_id", "CohortID", "cusip", "CUSIP"),
            "native_sector": _first_nonblank_text(frame, "native_sector", "source_sector", "sector"),
            "broad_holder_class": broad_holder.replace("", pd.NA),
            "tdcsim_holder": tdcsim_holder.replace("", pd.NA),
            "security_type": security_type,
            "component": pd.NA,
            "issue_date": issue_date,
            "maturity_date": maturity_date,
            "exposure_start": exposure_start,
            "exposure_end": exposure_end,
            "days_held": days_held,
            "exposure_fraction": days_held / float(quarter_days),
            "face_begin": face_begin,
            "face_end": face_end,
            "exposure_base": adjusted_principal.where(security_type.eq("TIPS"), face_value),
            "coupon_rate_decimal": coupon_rate,
            "issue_price_ratio": issue_price_ratio,
            "issue_yield_decimal": issue_yield,
            "fixed_spread": fixed_spread,
            "index_ratio_start": index_ratio_start,
            "index_ratio_end": index_ratio_end,
            "ref_cpi_start": ref_cpi_start,
            "ref_cpi_end": ref_cpi_end,
            "derivation_status": pd.NA,
            "source_status": _text_from_columns(frame, "source_status").replace("", _DEFAULT_SOURCE_STATUS),
            "interest_amount": np.nan,
            "excluded_from_default_canonical": residual_holder,
        },
        index=frame.index,
    )

    component_frames: list[pd.DataFrame] = []
    maturity_category = _text_from_columns(frame, "MaturityCategory", "maturity_category").str.lower()
    original_maturity = _numeric_from_columns(frame, "OriginalMaturityYears", "original_maturity_years", default=np.nan)
    bill_like = (
        security_type.eq("Fixed")
        & (
            maturity_category.eq("bills")
            | (
                (coupon_rate <= TGA_FLOOR_TOLERANCE)
                & original_maturity.notna()
                & (original_maturity <= 1.0 + TGA_FLOOR_TOLERANCE)
            )
        )
    )
    issue_proceeds_for_bills = issue_proceeds.where(issue_proceeds.notna(), face_value * issue_price_ratio)
    discount = face_value - issue_proceeds_for_bills
    life_days = ((maturity_date - issue_date).dt.days + 1).where(maturity_date.notna() & issue_date.notna(), 0)
    bill_mask = bill_like & (discount > TGA_FLOOR_TOLERANCE) & (life_days > 0)
    if bill_mask.any():
        component_frames.append(
            _component_frame(
                base,
                bill_mask,
                component="bill_discount_amortization",
                derivation_status="stock_only_bill_discount_life_fraction",
                interest_amount=discount * days_held / life_days.replace(0, np.nan),
                coupon_rate_decimal=0.0,
            )
        )

    fixed_mask = (
        security_type.eq("Fixed")
        & ~bill_like
        & (coupon_rate > TGA_FLOOR_TOLERANCE)
        & (face_value > TGA_FLOOR_TOLERANCE)
    )
    if fixed_mask.any():
        component_frames.append(
            _component_frame(
                base,
                fixed_mask,
                component="fixed_coupon_accrual",
                derivation_status="stock_only_fixed_coupon_act365",
                interest_amount=face_value * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL,
            )
        )

    accrued_frn = _numeric_from_columns(frame, "AccruedInterest_FRN", "accrued_interest_frn", default=0.0)
    frn_mask = security_type.eq("FRN") & (accrued_frn > TGA_FLOOR_TOLERANCE)
    if frn_mask.any():
        component_frames.append(
            _component_frame(
                base,
                frn_mask,
                component="frn_accrued_interest_pass_through",
                derivation_status="snapshot_frn_accrued_interest_pass_through",
                interest_amount=accrued_frn,
                coupon_rate_decimal=0.0,
            )
        )

    tips_mask = (
        security_type.eq("TIPS")
        & (coupon_rate > TGA_FLOOR_TOLERANCE)
        & (adjusted_principal > TGA_FLOOR_TOLERANCE)
    )
    if tips_mask.any():
        component_frames.append(
            _component_frame(
                base,
                tips_mask,
                component="tips_coupon_accrual",
                derivation_status="stock_only_tips_coupon_on_adjusted_principal",
                interest_amount=adjusted_principal * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL,
            )
        )

    if not component_frames:
        return pd.DataFrame(columns=DETAIL_COLUMNS)
    return pd.concat(component_frames, ignore_index=True).loc[:, DETAIL_COLUMNS]


def _component_frame(
    base: pd.DataFrame,
    mask: pd.Series,
    *,
    component: str,
    derivation_status: str,
    interest_amount: pd.Series,
    coupon_rate_decimal: float | None = None,
) -> pd.DataFrame:
    out = base.loc[mask].copy()
    out["component"] = component
    out["derivation_status"] = derivation_status
    out["interest_amount"] = pd.to_numeric(interest_amount.loc[mask], errors="coerce")
    if coupon_rate_decimal is not None:
        out["coupon_rate_decimal"] = coupon_rate_decimal
    return out


def _value_from_columns(frame: pd.DataFrame, *columns: str):
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return pd.Series(pd.NA, index=frame.index)


def _numeric_from_columns(frame: pd.DataFrame, *columns: str, default=np.nan) -> pd.Series:
    values = _value_from_columns(frame, *columns)
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.fillna(default) if not pd.isna(default) else numeric


def _text_from_columns(frame: pd.DataFrame, *columns: str) -> pd.Series:
    values = _value_from_columns(frame, *columns)
    return values.where(values.notna(), "").astype(str).str.strip()


def _first_nonblank_text(frame: pd.DataFrame, *columns: str) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column].where(frame[column].notna(), "").astype(str).str.strip()
        result = result.where(result.astype(str).str.strip().ne(""), candidate)
    return result.replace("", pd.NA)


def _frame_sha256(frame: pd.DataFrame | None) -> str:
    if frame is None or frame.empty:
        return "0" * 64
    stable = frame.copy()
    stable = stable.reindex(sorted(stable.columns), axis=1)
    payload = stable.to_csv(index=False, lineterminator="\n").encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _first_non_null_value(values: pd.Series):
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else pd.NA


def _quarter_bounds(quarter: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    period = pd.Period(str(quarter), freq="Q")
    quarter_start = period.start_time.normalize()
    quarter_end = period.end_time.normalize()
    quarter_days = int((quarter_end - quarter_start).days) + 1
    return quarter_start, quarter_end, quarter_days


def _quarter_start(quarter: str) -> pd.Timestamp:
    return pd.Period(str(quarter), freq="Q").start_time.normalize()


def _quarter_end_exclusive(quarter: str) -> pd.Timestamp:
    return (pd.Period(str(quarter), freq="Q") + 1).start_time.normalize()


def _overlapping_quarters(
    start_date: pd.Timestamp,
    end_date_exclusive: pd.Timestamp,
    start_quarter: str | None,
    end_quarter: str | None,
) -> list[pd.Period]:
    first = start_date.to_period("Q")
    last = (end_date_exclusive - pd.Timedelta(days=1)).to_period("Q")
    if start_quarter is not None:
        first = max(first, pd.Period(str(start_quarter), freq="Q"))
    if end_quarter is not None:
        last = min(last, pd.Period(str(end_quarter), freq="Q"))
    if last < first:
        return []
    return list(pd.period_range(first, last, freq="Q"))


def _semiannual_coupon_dates(
    issue_date: pd.Timestamp,
    first_coupon_date: pd.Timestamp,
    contract_end: pd.Timestamp,
) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    current = first_coupon_date.normalize()
    for _ in range(240):
        if current > contract_end:
            break
        if current > issue_date:
            dates.append(current)
        if current == contract_end:
            break
        current = current + DateOffset(months=6)
    if not dates or dates[-1] != contract_end:
        reverse: list[pd.Timestamp] = []
        current = contract_end.normalize()
        for _ in range(240):
            if current <= issue_date:
                break
            reverse.append(current)
            current = current - DateOffset(months=6)
        dates = sorted(reverse)
    return dates


def _semiannual_coupon_fractions(issue_date: pd.Timestamp, payment_dates: list[pd.Timestamp]) -> list[float]:
    if not payment_dates:
        return []
    first_coupon = payment_dates[0]
    previous_coupon = first_coupon - DateOffset(months=6)
    full_period_days = max(1, int((first_coupon - previous_coupon).days))
    first_stub_days = max(0, int((first_coupon - issue_date).days))
    first_fraction = first_stub_days / float(full_period_days)
    return [first_fraction] + [1.0] * (len(payment_dates) - 1)


def _terminal_clean_carrying_value(price_per100: float, coupon_rate: float, effective_yield: float, period_fractions: list[float]) -> float:
    cv = float(price_per100)
    for period_fraction in period_fractions:
        cv += cv * (effective_yield / 2.0) * period_fraction - 100.0 * (coupon_rate / 2.0) * period_fraction
    return cv


def _solve_effective_annual_yield(price_per100: float, coupon_rate: float, period_fractions: list[float]) -> float:
    if price_per100 <= 0.0 or not period_fractions:
        return np.nan

    def objective(yield_value: float) -> float:
        return _terminal_clean_carrying_value(price_per100, coupon_rate, yield_value, period_fractions) - 100.0

    low = -0.99
    high = 2.0
    low_value = objective(low)
    high_value = objective(high)
    if pd.isna(low_value) or pd.isna(high_value) or low_value * high_value > 0.0:
        return np.nan
    for _ in range(200):
        mid = (low + high) / 2.0
        mid_value = objective(mid)
        if abs(mid_value) <= 1.0e-12:
            return mid
        if low_value * mid_value <= 0.0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    return (low + high) / 2.0


def _exposure_window(
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
    issue_date: pd.Timestamp | pd.NaT,
    maturity_date: pd.Timestamp | pd.NaT,
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    exposure_start = quarter_start if pd.isna(issue_date) else max(quarter_start, issue_date.normalize())
    exposure_end = quarter_end if pd.isna(maturity_date) else min(quarter_end, maturity_date.normalize())
    if exposure_end < exposure_start:
        return exposure_start, exposure_end, 0
    return exposure_start, exposure_end, int((exposure_end - exposure_start).days) + 1


def _base_record(
    *,
    quarter: str,
    row: pd.Series,
    security_type: str,
    issue_date: pd.Timestamp | pd.NaT,
    maturity_date: pd.Timestamp | pd.NaT,
    quarter_start: pd.Timestamp,
    quarter_end: pd.Timestamp,
    exposure_start: pd.Timestamp,
    exposure_end: pd.Timestamp,
    days_held: int,
    quarter_days: int,
    face_value: float,
    adjusted_principal: float,
) -> dict[str, object]:
    broad_holder = str(row.get("broad_holder_class", row.get("HolderType", row.get("holder_type", "")))).strip()
    holder_subbucket = str(row.get("HolderSubBucket", row.get("holder_subbucket", ""))).strip()
    if not broad_holder:
        broad_holder = pd.NA
    tdcsim_holder = row.get("tdcsim_holder")
    if pd.isna(tdcsim_holder) or str(tdcsim_holder).strip() == "":
        if broad_holder == _PRIVATE_PREFIX and holder_subbucket:
            tdcsim_holder = f"{_PRIVATE_PREFIX}:{holder_subbucket}"
        else:
            tdcsim_holder = broad_holder

    issue_price_ratio = _num(
        row.get("IssuePriceRatio", row.get("issue_price_ratio")),
        default=np.nan,
    )
    issue_proceeds = _num(
        row.get("IssueProceeds", row.get("issue_proceeds")),
        default=np.nan,
    )
    if pd.isna(issue_price_ratio) and face_value > TGA_FLOOR_TOLERANCE and pd.notna(issue_proceeds):
        issue_price_ratio = issue_proceeds / face_value

    index_ratio_start, index_ratio_end = _tips_index_ratios(row)
    ref_cpi_start, ref_cpi_end = _tips_reference_cpi(row, index_ratio_start, index_ratio_end)
    face_begin = 0.0 if pd.notna(issue_date) and issue_date.normalize() > quarter_start.normalize() else face_value
    face_end = 0.0 if pd.notna(maturity_date) and maturity_date.normalize() < quarter_end.normalize() else face_value

    return {
        "quarter": quarter,
        "cusip/cohort_id": _cusip_or_cohort_id(row),
        "native_sector": row.get("native_sector", row.get("source_sector", row.get("sector", pd.NA))),
        "broad_holder_class": broad_holder,
        "tdcsim_holder": tdcsim_holder,
        "security_type": security_type,
        "component": pd.NA,
        "issue_date": issue_date,
        "maturity_date": maturity_date,
        "exposure_start": exposure_start,
        "exposure_end": exposure_end,
        "days_held": days_held,
        "exposure_fraction": days_held / float(quarter_days) if quarter_days > 0 else np.nan,
        "face_begin": face_begin,
        "face_end": face_end,
        "exposure_base": adjusted_principal if security_type == "TIPS" else face_value,
        "coupon_rate_decimal": _num(row.get("CouponRate", row.get("coupon_rate")), default=0.0),
        "issue_price_ratio": issue_price_ratio,
        "issue_yield_decimal": _num(
            row.get("IssueYieldAtIssue", row.get("issue_yield_decimal", row.get("IssueYield", np.nan))),
            default=np.nan,
        ),
        "fixed_spread": _num(row.get("FixedSpread", row.get("fixed_spread")), default=np.nan),
        "index_ratio_start": index_ratio_start,
        "index_ratio_end": index_ratio_end,
        "ref_cpi_start": ref_cpi_start,
        "ref_cpi_end": ref_cpi_end,
        "derivation_status": pd.NA,
        "source_status": row.get("source_status", _DEFAULT_SOURCE_STATUS),
        "interest_amount": np.nan,
        "excluded_from_default_canonical": False,
    }


def _build_fixed_coupon_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    days_held: int,
    security_type: str,
    face_value: float,
) -> dict[str, object] | None:
    if security_type != "Fixed" or _is_bill_like_row(row):
        return None
    coupon_rate = _num(base_record["coupon_rate_decimal"], default=0.0)
    if coupon_rate <= TGA_FLOOR_TOLERANCE or face_value <= TGA_FLOOR_TOLERANCE:
        return None
    record = dict(base_record)
    record["component"] = "fixed_coupon_accrual"
    record["derivation_status"] = "stock_only_fixed_coupon_act365"
    record["interest_amount"] = face_value * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL
    return record


def _build_bill_discount_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    days_held: int,
    security_type: str,
    issue_date: pd.Timestamp | pd.NaT,
    maturity_date: pd.Timestamp | pd.NaT,
) -> dict[str, object] | None:
    if security_type != "Fixed" or not _is_bill_like_row(row):
        return None
    if pd.isna(issue_date) or pd.isna(maturity_date):
        return None
    face_value = _num(row.get("FaceValue", row.get("face_value")))
    issue_proceeds = _num(row.get("IssueProceeds", row.get("issue_proceeds")), default=np.nan)
    if pd.isna(issue_proceeds):
        issue_price_ratio = _num(row.get("IssuePriceRatio", row.get("issue_price_ratio")), default=np.nan)
        if face_value > TGA_FLOOR_TOLERANCE and pd.notna(issue_price_ratio):
            issue_proceeds = face_value * issue_price_ratio
    discount = face_value - issue_proceeds
    life_days = int((maturity_date.normalize() - issue_date.normalize()).days) + 1
    if discount <= TGA_FLOOR_TOLERANCE or life_days <= 0:
        return None
    record = dict(base_record)
    record["component"] = "bill_discount_amortization"
    record["coupon_rate_decimal"] = 0.0
    record["derivation_status"] = "stock_only_bill_discount_life_fraction"
    record["interest_amount"] = discount * days_held / float(life_days)
    return record


def _build_frn_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    security_type: str,
) -> dict[str, object] | None:
    if security_type != "FRN":
        return None
    accrued_interest = _num(row.get("AccruedInterest_FRN", row.get("accrued_interest_frn")), default=0.0)
    if accrued_interest <= TGA_FLOOR_TOLERANCE:
        return None
    record = dict(base_record)
    record["component"] = "frn_accrued_interest_pass_through"
    record["coupon_rate_decimal"] = 0.0
    record["derivation_status"] = "snapshot_frn_accrued_interest_pass_through"
    record["interest_amount"] = accrued_interest
    return record


def _build_tips_coupon_record(
    *,
    row: pd.Series,
    base_record: dict[str, object],
    days_held: int,
    security_type: str,
    adjusted_principal: float,
) -> dict[str, object] | None:
    if security_type != "TIPS":
        return None
    coupon_rate = _num(base_record["coupon_rate_decimal"], default=0.0)
    if coupon_rate <= TGA_FLOOR_TOLERANCE or adjusted_principal <= TGA_FLOOR_TOLERANCE:
        return None
    record = dict(base_record)
    record["component"] = "tips_coupon_accrual"
    record["derivation_status"] = "stock_only_tips_coupon_on_adjusted_principal"
    record["interest_amount"] = adjusted_principal * coupon_rate * days_held / DAYS_PER_YEAR_ACTUAL
    return record


def _tips_adjusted_principal(row: pd.Series, *, fallback_face: float) -> float:
    adjusted_principal = _num(row.get("AdjustedPrincipal", row.get("adjusted_principal")), default=np.nan)
    if pd.notna(adjusted_principal) and adjusted_principal > TGA_FLOOR_TOLERANCE:
        return adjusted_principal
    original_principal = _num(
        row.get("OriginalPrincipal", row.get("original_principal")),
        default=np.nan,
    )
    index_ratio = _num(row.get("IndexRatio", row.get("index_ratio")), default=np.nan)
    if pd.notna(original_principal) and pd.notna(index_ratio):
        derived = original_principal * index_ratio
        if derived > TGA_FLOOR_TOLERANCE:
            return derived
    return fallback_face


def _tips_index_ratios(row: pd.Series) -> tuple[float, float]:
    end_ratio = _num(row.get("IndexRatioEnd", row.get("IndexRatio", row.get("index_ratio_end", row.get("index_ratio")))), default=np.nan)
    start_ratio = _num(row.get("IndexRatioStart", row.get("index_ratio_start")), default=np.nan)
    if pd.isna(start_ratio):
        start_ratio = end_ratio
    return start_ratio, end_ratio


def _tips_reference_cpi(row: pd.Series, index_ratio_start: float, index_ratio_end: float) -> tuple[float, float]:
    start_cpi = _num(row.get("ReferenceCPI_Start", row.get("ref_cpi_start")), default=np.nan)
    end_cpi = _num(row.get("ReferenceCPI_End", row.get("ref_cpi_end")), default=np.nan)
    if pd.notna(start_cpi) or pd.notna(end_cpi):
        if pd.isna(start_cpi):
            start_cpi = end_cpi
        if pd.isna(end_cpi):
            end_cpi = start_cpi
        return start_cpi, end_cpi
    base_cpi = _num(
        row.get("ReferenceCPI_Issue", row.get("reference_cpi_issue")),
        default=np.nan,
    )
    if pd.isna(base_cpi):
        return np.nan, np.nan
    start_cpi = base_cpi * index_ratio_start if pd.notna(index_ratio_start) else np.nan
    end_cpi = base_cpi * index_ratio_end if pd.notna(index_ratio_end) else np.nan
    return start_cpi, end_cpi


def _cusip_or_cohort_id(row: pd.Series):
    for column in ("cusip", "CUSIP", "cohort_id", "CohortID"):
        value = row.get(column, pd.NA)
        if pd.notna(value) and str(value).strip():
            return str(value)
    return pd.NA


def _is_bill_like_row(row: pd.Series) -> bool:
    security_type = str(row.get("SecurityType", row.get("security_type", ""))).strip()
    if security_type != "Fixed":
        return False
    maturity_category = str(row.get("MaturityCategory", row.get("maturity_category", ""))).strip().lower()
    if maturity_category == "bills":
        return True
    coupon_rate = _num(row.get("CouponRate", row.get("coupon_rate")), default=0.0)
    original_maturity_years = _num(
        row.get("OriginalMaturityYears", row.get("original_maturity_years")),
        default=np.nan,
    )
    return (
        coupon_rate <= TGA_FLOOR_TOLERANCE
        and pd.notna(original_maturity_years)
        and original_maturity_years <= 1.0 + TGA_FLOOR_TOLERANCE
    )


def _tips_coupon_cash_at(
    cusip: str,
    coupon_date: pd.Timestamp,
    term: pd.Series,
    stock_by_cusip: pd.DataFrame,
    cpi_by_date: pd.Series,
) -> float:
    coupon_rate = _num(term.get("coupon_rate_decimal"), default=0.0)
    base_cpi = _num(term.get("ref_cpi_on_dated_date"), default=np.nan)
    if coupon_rate <= TGA_FLOOR_TOLERANCE or pd.isna(base_cpi) or base_cpi <= 0.0:
        return 0.0
    remaining_par = _tips_remaining_par_at(cusip, coupon_date, term, stock_by_cusip, allow_maturity_day=True)
    ref_cpi = cpi_by_date.get(coupon_date, np.nan)
    if remaining_par <= TGA_FLOOR_TOLERANCE or pd.isna(ref_cpi):
        return 0.0
    return float(remaining_par) * (float(ref_cpi) / float(base_cpi)) * coupon_rate / 2.0


def _tips_coupon_accrued_interest_at(
    cusip: str,
    date: pd.Timestamp,
    term: pd.Series,
    stock_by_cusip: pd.DataFrame,
    cpi_by_date: pd.Series,
) -> float:
    coupon_rate = _num(term.get("coupon_rate_decimal"), default=0.0)
    base_cpi = _num(term.get("ref_cpi_on_dated_date"), default=np.nan)
    if coupon_rate <= TGA_FLOOR_TOLERANCE or pd.isna(base_cpi) or base_cpi <= 0.0:
        return 0.0
    remaining_par = _tips_remaining_par_at(cusip, date, term, stock_by_cusip, allow_maturity_day=False)
    ref_cpi = cpi_by_date.get(date, np.nan)
    previous_coupon, next_coupon = _tips_coupon_bracket(date, term)
    if (
        remaining_par <= TGA_FLOOR_TOLERANCE
        or pd.isna(ref_cpi)
        or previous_coupon is None
        or next_coupon is None
    ):
        return 0.0
    elapsed_days = int((date - previous_coupon).days)
    period_days = int((next_coupon - previous_coupon).days)
    if elapsed_days <= 0 or period_days <= 0:
        return 0.0
    return (
        float(remaining_par)
        * (float(ref_cpi) / float(base_cpi))
        * coupon_rate
        / 2.0
        * elapsed_days
        / float(period_days)
    )


def _tips_remaining_par_at(
    cusip: str,
    date: pd.Timestamp,
    term: pd.Series,
    stock_by_cusip: pd.DataFrame,
    *,
    allow_maturity_day: bool,
) -> float:
    maturity_date = term.get("maturity_date")
    if pd.notna(maturity_date):
        maturity_date = pd.Timestamp(maturity_date).normalize()
        if date > maturity_date or (date == maturity_date and not allow_maturity_day):
            return 0.0
    rows = stock_by_cusip.loc[
        stock_by_cusip["cusip"].astype(str).eq(str(cusip))
        & (pd.to_datetime(stock_by_cusip["record_date"], errors="coerce") <= date)
    ]
    if rows.empty:
        return 0.0
    return _num(rows.iloc[-1].get("remaining_par_mil"), default=0.0)


def _tips_coupon_dates_between(
    cusip: str,
    start_exclusive: pd.Timestamp,
    end_inclusive: pd.Timestamp,
    term: pd.Series,
) -> list[pd.Timestamp]:
    del cusip
    maturity_date = term.get("maturity_date")
    if pd.isna(maturity_date):
        return []
    maturity_date = pd.Timestamp(maturity_date).normalize()
    month = int(maturity_date.month)
    day = int(maturity_date.day)
    alternate_month = month - 6 if month > 6 else month + 6
    dates: list[pd.Timestamp] = []
    for year in range(int(start_exclusive.year) - 1, int(end_inclusive.year) + 2):
        for coupon_month in [month, alternate_month]:
            coupon_day = min(day, calendar.monthrange(year, coupon_month)[1])
            coupon_date = pd.Timestamp(year=year, month=coupon_month, day=coupon_day)
            if start_exclusive < coupon_date <= end_inclusive and coupon_date <= maturity_date:
                dates.append(coupon_date)
    return sorted(set(dates))


def _tips_coupon_bracket(date: pd.Timestamp, term: pd.Series) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    maturity_date = term.get("maturity_date")
    if pd.isna(maturity_date):
        return None, None
    maturity_date = pd.Timestamp(maturity_date).normalize()
    if date >= maturity_date:
        return None, None
    month = int(maturity_date.month)
    day = int(maturity_date.day)
    alternate_month = month - 6 if month > 6 else month + 6
    anchors: list[pd.Timestamp] = []
    for year in range(int(date.year) - 2, int(date.year) + 3):
        for coupon_month in [month, alternate_month]:
            coupon_day = min(day, calendar.monthrange(year, coupon_month)[1])
            coupon_date = pd.Timestamp(year=year, month=coupon_month, day=coupon_day)
            if coupon_date <= maturity_date:
                anchors.append(coupon_date)
    anchors = sorted(set(anchors))
    previous_coupon = max((coupon_date for coupon_date in anchors if coupon_date <= date), default=None)
    next_coupon = min((coupon_date for coupon_date in anchors if coupon_date > date), default=None)
    return previous_coupon, next_coupon


def _coupon_anchor_dates(
    pay_date_1,
    pay_date_2,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    anchors = []
    for value in [pay_date_1, pay_date_2]:
        parsed = _parse_month_day(value)
        if parsed is not None:
            anchors.append(parsed)
    if not anchors:
        return []
    if len(anchors) == 1:
        month, day = anchors[0]
        prior_month = month - 6
        prior_year_shift = 0
        if prior_month <= 0:
            prior_month += 12
            prior_year_shift = -1
        anchors.append((prior_month, day if prior_year_shift == 0 else day))
    dates: list[pd.Timestamp] = []
    for year in range(int(start.year) - 1, int(end.year) + 2):
        for month, day in anchors:
            last_day = calendar.monthrange(year, month)[1]
            dates.append(pd.Timestamp(year=year, month=month, day=min(day, last_day)))
    return sorted(set(dates))


def _parse_month_day(value) -> tuple[int, int] | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split("/")
    if len(parts) < 2:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return None
        return int(parsed.month), int(parsed.day)
    try:
        month = int(parts[0])
        day = int(parts[1])
    except ValueError:
        return None
    if month < 1 or month > 12 or day < 1:
        return None
    return month, day


def _coupon_factor(start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp, coupon_dates: list[pd.Timestamp]) -> float:
    if pd.isna(start_exclusive) or pd.isna(end_inclusive) or end_inclusive <= start_exclusive:
        return 0.0
    factor = 0.0
    for period_start, period_end in zip(coupon_dates[:-1], coupon_dates[1:]):
        if period_end <= start_exclusive or period_start >= end_inclusive:
            continue
        overlap_start = max(start_exclusive, period_start)
        overlap_end = min(end_inclusive, period_end)
        overlap_days = int((overlap_end - overlap_start).days)
        period_days = int((period_end - period_start).days)
        if overlap_days <= 0 or period_days <= 0:
            continue
        factor += 0.5 * overlap_days / float(period_days)
    return float(factor)


def _calendar_midpoint(start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(start_exclusive) or pd.isna(end_inclusive):
        return pd.NaT
    days = int((end_inclusive - start_exclusive).days)
    if days <= 0:
        return end_inclusive
    return start_exclusive + pd.Timedelta(days=days // 2)


def _frn_discount_to_index_pct(discount_rate_pct, term_days) -> float:
    discount = _num(discount_rate_pct, default=np.nan)
    days = _num(term_days, default=np.nan)
    if pd.isna(discount) or pd.isna(days):
        return np.nan
    return round(discount / (1.0 - (discount / 100.0) * days / 360.0), 9)


def _frn_count_by_cusip(frame: pd.DataFrame | None, date_column: str, output_column: str) -> pd.DataFrame:
    if frame is None or frame.empty or "cusip" not in frame.columns:
        return pd.DataFrame(columns=["cusip", output_column])
    working = frame.copy()
    working["cusip"] = working["cusip"].astype(str).str.strip()
    if date_column in working.columns:
        working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
        working = working.loc[working[date_column].notna()].copy()
    return (
        working.groupby("cusip", sort=False, dropna=False)
        .size()
        .rename(output_column)
        .reset_index()
    )


def _to_timestamp(value) -> pd.Timestamp | pd.NaT:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def _num(value, *, default=0.0):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return default
    return float(converted)


def _optional_mil(value):
    converted = pd.to_numeric(value, errors="coerce")
    if pd.isna(converted):
        return np.nan
    return float(converted) / 1_000_000.0


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


__all__ = [
    "BILL_INTEREST_FLOW_DETAIL_COLUMNS",
    "BILL_LOT_CONSERVATION_COLUMNS",
    "DETAIL_COLUMNS",
    "FIXED_COUPON_INTEREST_RECONCILIATION_COLUMNS",
    "FIXED_COUPON_MONTHLY_DETAIL_COLUMNS",
    "FIXED_COUPON_PRINCIPAL_ADJUSTMENT_COLUMNS",
    "FRN_CUSIP_COVERAGE_COLUMNS",
    "FRN_DAILY_INDEX_PATH_COLUMNS",
    "FRN_DAILY_INDEX_VALIDATION_COLUMNS",
    "FRN_INTEREST_DAILY_DETAIL_COLUMNS",
    "FRN_INTEREST_RECONCILIATION_COLUMNS",
    "FRN_PRINCIPAL_RECONCILIATION_COLUMNS",
    "INTEREST_COMPONENT_RECONCILIATION_COLUMNS",
    "INTEREST_COMPONENT_CERTIFICATION_COLUMNS",
    "INTEREST_SCOPE_CERTIFICATION_COLUMNS",
    "NONBILL_DISCOUNT_PREMIUM_DETAIL_COLUMNS",
    "OFFICIAL_INTEREST_SCOPE_BRIDGE_COLUMNS",
    "SECTOR_INTEREST_ALLOCATION_COLUMNS",
    "TIPS_INFLATION_MONTHLY_DETAIL_COLUMNS",
    "TIPS_INFLATION_RECONCILIATION_COLUMNS",
    "TREASURY_INTEREST_EXPENSE_DIAGNOSTIC_COLUMNS",
    "aggregate_stock_only_interest",
    "align_stock_only_interest_proxy",
    "build_bill_interest_flow_detail",
    "build_bill_lot_conservation",
    "build_fixed_coupon_interest_flow_detail",
    "build_fixed_coupon_interest_reconciliation",
    "build_fixed_coupon_principal_adjustments",
    "build_frn_cusip_coverage",
    "build_frn_daily_index_backcast_validation",
    "build_frn_daily_index_path",
    "build_frn_interest_flow_detail",
    "build_frn_interest_reconciliation",
    "build_frn_principal_reconciliation",
    "build_interest_component_reconciliation",
    "build_interest_component_certification",
    "build_interest_scope_certification",
    "build_nonbill_discount_premium_flow_detail",
    "build_official_interest_scope_bridge",
    "build_quarterly_interest_detail",
    "build_sector_interest_allocation",
    "build_sector_interest_totals",
    "build_tips_inflation_compensation_flow_detail",
    "build_tips_inflation_compensation_reconciliation",
    "build_treasury_interest_expense_diagnostic",
]
