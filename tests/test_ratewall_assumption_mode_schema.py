from __future__ import annotations

import pandas as pd

from ratewall_assumption_mode_schema import validate_route_support_rows


def _support_row(**overrides: str) -> dict[str, str]:
    row = {
        "support_row_id": "row1",
        "route_component_id": "tdcsim_private_bucket_route_sensitivity",
        "object_family": "stock_interest_quarter_end",
        "measurement_stage": "holder_stock",
        "evidence_tier": "bounded_proxy",
        "mapping_burden": "requires_unobserved_actor_split",
        "assumption_status": "bounded_assumption",
        "admissible_use": "assumption_mode_support_ledger;assumption_mode_sensitivity",
        "blocked_use": (
            "source_backed_private_bucket_split;canonical_tdc_math;"
            "evidence_mode;final_current_demand;holder_allocation;"
            "pricing_incidence_welfare_tax_mpc;prior_narrowing"
        ),
        "source_backed_private_bucket_split_status": (
            "not_source_backed_private_bucket_split"
        ),
        "source_backed_private_bucket_split_row": "false",
        "canonical_tdc_math_change": "false",
        "evidence_mode_enabled": "false",
        "current_demand_eligible": "false",
        "holder_allocation_enabled": "false",
        "pricing_output_enabled": "false",
        "incidence_claim_enabled": "false",
        "welfare_claim_enabled": "false",
        "tax_output_enabled": "false",
        "mpc_output_enabled": "false",
        "prior_narrowing_allowed": "false",
    }
    row.update(overrides)
    return row


def test_validate_route_support_rows_accepts_bounded_sidecar_rows() -> None:
    report = validate_route_support_rows(pd.DataFrame([_support_row()]))

    assert report.passed
    assert report.validation_status == "pass"
    assert report.failure_reasons == ()


def test_validate_route_support_rows_rejects_bounded_evidence_promotion() -> None:
    report = validate_route_support_rows(
        pd.DataFrame(
            [
                _support_row(
                    admissible_use="canonical_tdcsim_input",
                    evidence_mode_enabled="true",
                )
            ]
        )
    )

    assert not report.passed
    assert any(
        "non_promoting_tier_marked_canonical_input" in reason
        for reason in report.failure_reasons
    )
    assert any(
        "non_promoting_tier_enabled:evidence_mode_enabled" in reason
        for reason in report.failure_reasons
    )
