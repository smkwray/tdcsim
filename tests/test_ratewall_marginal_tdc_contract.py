import pandas as pd

from ratewall_marginal_tdc_contract import CLAIM_BOUNDARY, validate_ratewall_marginal_tdc_summary


def test_ratewall_marginal_tdc_contract_validates_support_identity() -> None:
    summary = pd.DataFrame([_summary_row()])

    assert validate_ratewall_marginal_tdc_summary(summary) == {"status": "pass", "rows": 1}
    summary.loc[0, "legacy_chi_support_diagnostic_bil"] = 0.6
    assert validate_ratewall_marginal_tdc_summary(summary)["status"] == "fail"


def test_retired_chi_support_is_not_published_as_a_neutral_headline_field() -> None:
    """The retired beta-chi object must not look like the selected one.

    It previously shipped as `marginal_tdc_support_bil` with a neutral
    `support_formula`, in the same row that declared chi `retired_not_selected`.
    A consumer reading the obvious field got the retired construction, which also
    reintroduced the interest component the split declares excluded.
    """

    summary = pd.DataFrame([_summary_row()])

    assert "marginal_tdc_support_bil" not in summary.columns
    assert "support_formula" not in summary.columns
    assert summary.loc[0, "chi_selected_status"] == "retired_not_selected"
    assert bool(summary.loc[0, "legacy_chi_support_eligible_for_main_ratio"]) is False
    assert validate_ratewall_marginal_tdc_summary(summary) == {"status": "pass", "rows": 1}


def test_ratewall_contract_rejects_legacy_support_marked_ratio_eligible() -> None:
    summary = pd.DataFrame([_summary_row()])
    summary.loc[0, "legacy_chi_support_eligible_for_main_ratio"] = True

    result = validate_ratewall_marginal_tdc_summary(summary)

    assert result["status"] == "fail"
    assert "ineligible for the main ratio" in result["failure_reason"]


def test_ratewall_marginal_tdc_contract_fails_closed_without_split_fields() -> None:
    summary = pd.DataFrame([_summary_row()])
    summary = summary.drop(columns=["delta_tdc_ex_overlap_non_interest_admissible_bil"])

    result = validate_ratewall_marginal_tdc_summary(summary)

    assert result["status"] == "fail"
    assert "missing fields" in str(result["failure_reason"])


def test_ratewall_marginal_tdc_contract_accepts_fiscal_injection_path() -> None:
    summary = pd.DataFrame(
        [
            _summary_row(
                scenario_state_set_id="flooded_2028_v1",
                object_id="TDC_FISCAL_INJECTION_2028",
                shock_path_id="fiscal_injection_2028_v1",
                shock_bps_year=0,
                state_id="pre_flood_state::2028",
                state_kind="scenario_state",
                state_period="2028",
                period="2028",
                period_start="2028-01-01",
                period_end="2029-01-01",
                horizon="annual_h1_fiscal_injection",
                demand_conversion_case="pre_beta_pair",
                tdc_change_baseline_bil=10.0,
                tdc_change_shock_bil=3013.0,
                delta_tdc_change_bil=3003.0,
                overlap_baseline_bil=2.0,
                overlap_shock_bil=5.0,
                delta_overlap_bil=3.0,
                tdc_change_ex_overlap_baseline_bil=8.0,
                tdc_change_ex_overlap_shock_bil=3008.0,
                delta_tdc_ex_overlap_bil=3000.0,
                interest_excluded=1000.0,
                admissible=2000.0,
                beta=1.0,
                chi=1.0,
                beta_assumption_id="ratewall_side_conversion_pending",
                chi_assumption_id="ratewall_side_conversion_pending",
                rate_shock_only_status="not_applicable_fiscal_injection_no_rate_shock",
            )
        ]
    )

    assert validate_ratewall_marginal_tdc_summary(summary) == {"status": "pass", "rows": 1}


def _summary_row(
    *,
    scenario_state_set_id: str = "state_set",
    object_id: str = "RW_M_PLUS_100BP_YEAR",
    shock_path_id: str = "plus_100bp_year",
    shock_bps_year: float = 100,
    state_id: str = "state",
    state_kind: str = "current_state",
    state_period: str = "2026-01-01",
    period: str = "2027-01-01",
    period_start: str = "2026-01-01",
    period_end: str = "2027-01-01",
    horizon: str = "annual_h1_100bp_year",
    demand_conversion_case: str = "central",
    tdc_change_baseline_bil: float = 10.0,
    tdc_change_shock_bil: float = 13.0,
    delta_tdc_change_bil: float = 3.0,
    overlap_baseline_bil: float = 2.0,
    overlap_shock_bil: float = 3.0,
    delta_overlap_bil: float = 1.0,
    tdc_change_ex_overlap_baseline_bil: float = 8.0,
    tdc_change_ex_overlap_shock_bil: float = 10.0,
    delta_tdc_ex_overlap_bil: float = 2.0,
    interest_excluded: float = 0.75,
    admissible: float = 1.25,
    beta: float = 0.5,
    chi: float = 0.4,
    beta_assumption_id: str = "beta_fixture",
    chi_assumption_id: str = "chi_fixture",
    rate_shock_only_status: str = "pass",
) -> dict:
    beta_times_chi = beta * chi
    support = delta_tdc_ex_overlap_bil * beta_times_chi
    return {
        "schema_version": "tdcsim_cbo_marginal_tdc_pair_v1",
        "tdc_deposit_creation_split_schema_version": "tdc_deposit_creation_split_v1",
        "contract_version": "0.4.0",
        "pair_id": "pair",
        "scenario_state_set_id": scenario_state_set_id,
        "object_id": object_id,
        "shock_path_id": shock_path_id,
        "shock_bps_year": shock_bps_year,
        "state_id": state_id,
        "state_kind": state_kind,
        "state_period": state_period,
        "scenario_id": "scenario",
        "state_fingerprint_sha256": "a" * 64,
        "state_component_inventory_sha256": "b" * 64,
        "period": period,
        "period_start": period_start,
        "period_end": period_end,
        "horizon": horizon,
        "demand_conversion_case": demand_conversion_case,
        "baseline_run_id": "base",
        "shock_run_id": "shock",
        "tdc_change_baseline_bil": tdc_change_baseline_bil,
        "tdc_change_shock_bil": tdc_change_shock_bil,
        "delta_tdc_change_bil": delta_tdc_change_bil,
        "overlap_baseline_bil": overlap_baseline_bil,
        "overlap_shock_bil": overlap_shock_bil,
        "delta_overlap_bil": delta_overlap_bil,
        "tdc_change_ex_overlap_baseline_bil": tdc_change_ex_overlap_baseline_bil,
        "tdc_change_ex_overlap_shock_bil": tdc_change_ex_overlap_shock_bil,
        "delta_tdc_ex_overlap_bil": delta_tdc_ex_overlap_bil,
        "delta_tdc_ex_overlap_interest_driven_excluded_bil": interest_excluded,
        "delta_tdc_ex_overlap_non_interest_admissible_bil": admissible,
        "delta_tdc_ex_overlap_split_remainder_bil": 0.0,
        "delta_tdc_ex_overlap_reconciled_bil": interest_excluded + admissible,
        "tdc_materialized_deposit_stock_admissible_bil": admissible * beta,
        "tdc_materialized_deposit_stock_interest_excluded_bil": interest_excluded * beta,
        "tdc_income_addendum_full_level_rate": 0.035,
        "tdc_income_addendum_gross_interest_bil": admissible * beta * 0.035,
        "tdc_income_addendum_route_family": "tdc_income_from_tdcsim_marginal_deposit_stock",
        "tdc_income_addendum_admission_status": "admitted_split_non_interest_bucket",
        "tdc_income_addendum_collision_status": "pass_split_collision_excluded",
        "selected_support_formula": "admissible \u00d7 \u03b2 \u00d7 rate \u00d7 sfc_route_coefficients",
        "beta_assumption_id": beta_assumption_id,
        "beta": beta,
        "chi_assumption_id": chi_assumption_id,
        "chi": chi,
        "beta_times_chi": beta_times_chi,
        "tdc_amount_basis": "pre_beta_ex_overlap_delta",
        "legacy_support_formula": "delta_tdc_ex_overlap_bil * beta * chi",
        "legacy_chi_support_diagnostic_bil": support,
        "legacy_chi_support_eligible_for_main_ratio": False,
        "chi_selected_status": "retired_not_selected",
        "same_state_status": "pass",
        "rate_shock_only_status": rate_shock_only_status,
        "shock_path_validation_status": "pass",
        "period_alignment_status": "pass",
        "overlap_identity_status": "pass",
        "component_identity_status": "pass",
        "route_identity_status": "pass",
        "support_identity_status": "pass",
        "state_manifest_status": "pass",
        "contract_ingest_status": "ready_for_ratewall_assumption_mode_ingest",
        "failure_reason": "",
        "assumption_mode": True,
        "evidence_mode_enabled": False,
        "raw_rate_shock_enabled": False,
        "named_marginal_shock_path_enabled": True,
        "tdcsim_channel_classifier_enabled": False,
        "enters_main_ratio_candidate": True,
        "canonical_ratio_entry": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
