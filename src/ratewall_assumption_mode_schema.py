from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


EVIDENCE_TIERS = {
    "direct_input",
    "source_backed_measurement",
    "source_backed_context",
    "bounded_proxy",
    "context_only",
    "assumption_only",
    "unresolved_residual",
}

ASSUMPTION_STATUSES = {
    "none_source_observed",
    "source_backed_but_not_route_identifying",
    "bounded_by_observed_context",
    "bounded_assumption",
    "prior_or_regularization_only",
    "mechanical_lambda_assumption",
    "speculative_sensitivity",
    "unresolved",
}

ADMISSIBLE_USES = {
    "canonical_tdcsim_input",
    "assumption_mode_support_ledger",
    "assumption_mode_sensitivity",
    "context_appendix",
    "diagnostic_only",
}

BLOCKED_USES = {
    "source_backed_private_bucket_split",
    "canonical_tdc_math",
    "evidence_mode",
    "final_current_demand",
    "holder_allocation",
    "pricing_incidence_welfare_tax_mpc",
    "prior_narrowing",
}

REQUIRED_SUPPORT_COLUMNS = {
    "support_row_id",
    "route_component_id",
    "object_family",
    "measurement_stage",
    "evidence_tier",
    "mapping_burden",
    "assumption_status",
    "admissible_use",
    "blocked_use",
    "source_backed_private_bucket_split_status",
    "source_backed_private_bucket_split_row",
    "canonical_tdc_math_change",
    "evidence_mode_enabled",
    "current_demand_eligible",
    "holder_allocation_enabled",
}

_FALSE_GUARDRAIL_COLUMNS = {
    "source_backed_private_bucket_split_row",
    "canonical_tdc_math_change",
    "evidence_mode_enabled",
    "current_demand_eligible",
    "holder_allocation_enabled",
    "pricing_output_enabled",
    "incidence_claim_enabled",
    "welfare_claim_enabled",
    "tax_output_enabled",
    "mpc_output_enabled",
    "prior_narrowing_allowed",
}

_NON_PROMOTING_TIERS = {
    "bounded_proxy",
    "context_only",
    "assumption_only",
    "unresolved_residual",
}


@dataclass(frozen=True)
class ValidationReport:
    validation_status: str
    row_count: int
    failure_reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.validation_status == "pass"


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _tokens(value: object) -> set[str]:
    return {token.strip() for token in str(value).split(";") if token.strip()}


def validate_route_support_rows(frame: pd.DataFrame) -> ValidationReport:
    """Validate RateWall Assumption Mode support rows.

    This validator is deliberately stricter for non-promoting evidence tiers:
    bounded, contextual, assumption-only, and residual rows must not unlock
    evidence mode, canonical math, holder allocation, current demand, pricing,
    incidence, welfare, tax, MPC, or prior-narrowing switches.
    """

    failures: list[str] = []
    missing = REQUIRED_SUPPORT_COLUMNS - set(frame.columns)
    if missing:
        failures.append("missing_columns:" + ",".join(sorted(missing)))
        return ValidationReport("fail", len(frame), tuple(failures))

    for index, row in frame.iterrows():
        row_id = str(row.get("support_row_id", index))
        tier = str(row.get("evidence_tier", ""))
        if tier not in EVIDENCE_TIERS:
            failures.append(f"{row_id}:unknown_evidence_tier:{tier}")
        if not str(row.get("measurement_stage", "")).strip():
            failures.append(f"{row_id}:missing_measurement_stage")

        admissible = _tokens(row.get("admissible_use", ""))
        blocked = _tokens(row.get("blocked_use", ""))
        unknown_admissible = admissible - ADMISSIBLE_USES
        if unknown_admissible:
            failures.append(
                f"{row_id}:unknown_admissible_use:{','.join(sorted(unknown_admissible))}"
            )
        if not blocked.intersection(BLOCKED_USES):
            failures.append(f"{row_id}:blocked_use_missing_guardrail_token")

        if tier in _NON_PROMOTING_TIERS:
            if "canonical_tdcsim_input" in admissible:
                failures.append(f"{row_id}:non_promoting_tier_marked_canonical_input")
            for column in _FALSE_GUARDRAIL_COLUMNS.intersection(frame.columns):
                if _is_true(row.get(column, "")):
                    failures.append(f"{row_id}:non_promoting_tier_enabled:{column}")
            if str(row.get("source_backed_private_bucket_split_status", "")) != (
                "not_source_backed_private_bucket_split"
            ):
                failures.append(f"{row_id}:private_split_status_not_fail_closed")

        if str(row.get("assumption_status", "")) not in ASSUMPTION_STATUSES:
            failures.append(
                f"{row_id}:unknown_assumption_status:{row.get('assumption_status', '')}"
            )

    return ValidationReport(
        "pass" if not failures else "fail",
        len(frame),
        tuple(failures),
    )
