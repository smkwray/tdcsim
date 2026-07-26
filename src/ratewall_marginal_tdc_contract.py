"""RateWall-facing validator for TDCSIM marginal TDC pair summaries."""

from __future__ import annotations

import pandas as pd


MARGINAL_TDC_SCHEMA_VERSION = "tdcsim_cbo_marginal_tdc_pair_v1"
TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION = "tdc_deposit_creation_split_v1"
CLAIM_BOUNDARY = "tdcsim_marginal_pair_assumption_mode_not_evidence_not_channel_classifier"
ALLOWED_OBJECT_IDS = {"RW_M_PLUS_100BP_YEAR", "TDC_FISCAL_INJECTION_2028"}
ALLOWED_SHOCK_PATH_IDS = {"plus_100bp_year", "fiscal_injection_2028_v1"}
REQUIRED_SUMMARY_FIELDS = {
    "schema_version",
    "tdc_deposit_creation_split_schema_version",
    "contract_version",
    "pair_id",
    "scenario_state_set_id",
    "object_id",
    "shock_path_id",
    "shock_bps_year",
    "state_id",
    "state_kind",
    "state_period",
    "scenario_id",
    "state_fingerprint_sha256",
    "state_component_inventory_sha256",
    "period",
    "period_start",
    "period_end",
    "horizon",
    "demand_conversion_case",
    "baseline_run_id",
    "shock_run_id",
    "tdc_change_baseline_bil",
    "tdc_change_shock_bil",
    "delta_tdc_change_bil",
    "overlap_baseline_bil",
    "overlap_shock_bil",
    "delta_overlap_bil",
    "tdc_change_ex_overlap_baseline_bil",
    "tdc_change_ex_overlap_shock_bil",
    "delta_tdc_ex_overlap_bil",
    "delta_tdc_ex_overlap_interest_driven_excluded_bil",
    "delta_tdc_ex_overlap_non_interest_admissible_bil",
    "delta_tdc_ex_overlap_split_remainder_bil",
    "delta_tdc_ex_overlap_reconciled_bil",
    "tdc_materialized_deposit_stock_admissible_bil",
    "tdc_materialized_deposit_stock_interest_excluded_bil",
    "tdc_income_addendum_full_level_rate",
    "tdc_income_addendum_gross_interest_bil",
    "tdc_income_addendum_route_family",
    "tdc_income_addendum_admission_status",
    "tdc_income_addendum_collision_status",
    "selected_support_formula",
    "legacy_support_formula",
    "beta_assumption_id",
    "beta",
    "chi_assumption_id",
    "chi",
    "beta_times_chi",
    "tdc_amount_basis",
    "legacy_chi_support_diagnostic_bil",
    "legacy_chi_support_eligible_for_main_ratio",
    "chi_selected_status",
    "same_state_status",
    "rate_shock_only_status",
    "shock_path_validation_status",
    "period_alignment_status",
    "overlap_identity_status",
    "component_identity_status",
    "route_identity_status",
    "support_identity_status",
    "state_manifest_status",
    "contract_ingest_status",
    "failure_reason",
    "assumption_mode",
    "evidence_mode_enabled",
    "raw_rate_shock_enabled",
    "named_marginal_shock_path_enabled",
    "tdcsim_channel_classifier_enabled",
    "enters_main_ratio_candidate",
    "canonical_ratio_entry",
    "claim_boundary",
}


def validate_ratewall_marginal_tdc_summary(summary: pd.DataFrame) -> dict[str, object]:
    """Validate that RateWall is receiving marginal support, not gross TDC exposure."""

    missing = sorted(REQUIRED_SUMMARY_FIELDS - set(summary.columns))
    if missing:
        return {"status": "fail", "failure_reason": f"missing fields: {missing}"}
    if set(summary["tdc_deposit_creation_split_schema_version"].astype(str)) != {TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION}:
        return {"status": "fail", "failure_reason": "split schema_version failed"}
    ex_identity = _series(summary, "delta_tdc_change_bil") - _series(summary, "delta_overlap_bil")
    if (ex_identity - _series(summary, "delta_tdc_ex_overlap_bil")).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "delta_tdc_ex_overlap identity failed"}
    beta_chi = _series(summary, "beta") * _series(summary, "chi")
    if (beta_chi - _series(summary, "beta_times_chi")).abs().max() > 1e-12:
        return {"status": "fail", "failure_reason": "beta_times_chi identity failed"}
    support = _series(summary, "delta_tdc_ex_overlap_bil") * beta_chi
    split = (
        _series(summary, "delta_tdc_ex_overlap_interest_driven_excluded_bil")
        + _series(summary, "delta_tdc_ex_overlap_non_interest_admissible_bil")
    )
    if (split - _series(summary, "delta_tdc_ex_overlap_reconciled_bil")).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "split reconciled identity failed"}
    if (split - _series(summary, "delta_tdc_ex_overlap_bil")).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "split ex-overlap identity failed"}
    if _series(summary, "delta_tdc_ex_overlap_split_remainder_bil").abs().max() > 1e-12:
        return {"status": "fail", "failure_reason": "split remainder failed"}
    if (
        _series(summary, "delta_tdc_ex_overlap_non_interest_admissible_bil") * _series(summary, "beta")
        - _series(summary, "tdc_materialized_deposit_stock_admissible_bil")
    ).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "admissible stock identity failed"}
    if (
        _series(summary, "delta_tdc_ex_overlap_interest_driven_excluded_bil") * _series(summary, "beta")
        - _series(summary, "tdc_materialized_deposit_stock_interest_excluded_bil")
    ).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "interest excluded stock identity failed"}
    if (
        _series(summary, "tdc_materialized_deposit_stock_admissible_bil")
        * _series(summary, "tdc_income_addendum_full_level_rate")
        - _series(summary, "tdc_income_addendum_gross_interest_bil")
    ).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "income addendum identity failed"}
    if set(summary["tdc_income_addendum_route_family"].astype(str)) != {"tdc_income_from_tdcsim_marginal_deposit_stock"}:
        return {"status": "fail", "failure_reason": "income addendum route family failed"}
    if set(summary["tdc_income_addendum_admission_status"].astype(str)) != {"admitted_split_non_interest_bucket"}:
        return {"status": "fail", "failure_reason": "income addendum admission status failed"}
    if set(summary["tdc_income_addendum_collision_status"].astype(str)) != {"pass_split_collision_excluded"}:
        return {"status": "fail", "failure_reason": "income addendum collision status failed"}
    if set(summary["selected_support_formula"].astype(str)) != {"admissible \u00d7 \u03b2 \u00d7 rate \u00d7 sfc_route_coefficients"}:
        return {"status": "fail", "failure_reason": "selected support formula failed"}
    if (support - _series(summary, "legacy_chi_support_diagnostic_bil")).abs().max() > 1e-7:
        return {"status": "fail", "failure_reason": "legacy chi diagnostic failed"}
    if set(summary["legacy_chi_support_eligible_for_main_ratio"].astype(str)) != {"False"}:
        return {"status": "fail", "failure_reason": "legacy chi support must be ineligible for the main ratio"}
    if set(summary["chi_selected_status"].astype(str)) != {"retired_not_selected"}:
        return {"status": "fail", "failure_reason": "chi selected status failed"}
    if set(summary["claim_boundary"].astype(str)) != {CLAIM_BOUNDARY}:
        return {"status": "fail", "failure_reason": "claim boundary failed"}
    if not set(summary["object_id"].astype(str)) <= ALLOWED_OBJECT_IDS:
        return {"status": "fail", "failure_reason": "object_id failed"}
    if not set(summary["shock_path_id"].astype(str)) <= ALLOWED_SHOCK_PATH_IDS:
        return {"status": "fail", "failure_reason": "shock_path_id failed"}
    if set(summary["tdc_amount_basis"].astype(str)) != {"pre_beta_ex_overlap_delta"}:
        return {"status": "fail", "failure_reason": "tdc_amount_basis failed"}
    if set(summary["legacy_support_formula"].astype(str)) != {"delta_tdc_ex_overlap_bil * beta * chi"}:
        return {"status": "fail", "failure_reason": "legacy_support_formula failed"}
    for field in (
        "same_state_status",
        "shock_path_validation_status",
        "route_identity_status",
        "state_manifest_status",
        "contract_ingest_status",
    ):
        if summary[field].astype(str).str.contains("fail", case=False, na=False).any():
            return {"status": "fail", "failure_reason": f"{field} failed"}
    rate_status = set(summary["rate_shock_only_status"].astype(str))
    if not rate_status <= {"pass", "not_applicable_fiscal_injection_no_rate_shock"}:
        return {"status": "fail", "failure_reason": "rate_shock_only_status failed"}
    return {"status": "pass", "rows": int(len(summary))}


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


__all__ = [
    "CLAIM_BOUNDARY",
    "MARGINAL_TDC_SCHEMA_VERSION",
    "REQUIRED_SUMMARY_FIELDS",
    "TDC_DEPOSIT_CREATION_SPLIT_SCHEMA_VERSION",
    "validate_ratewall_marginal_tdc_summary",
]
