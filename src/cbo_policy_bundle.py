"""CBO fiscal-incidence and net-interest bridge policy helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


FISCAL_INCIDENCE_SCHEMA_VERSION = "tdcsim_fiscal_incidence_policy_v1"
NET_INTEREST_BRIDGE_SCHEMA_VERSION = "tdcsim_net_interest_bridge_v1"

SIGNED_NET_PRIMARY_PROXY = "signed_net_primary_proxy"
EXPLICIT_SCENARIO_ASSUMPTION = "explicit_scenario_assumption"

DEFAULT_INCIDENCE_SPLITS = (
    ("central_99du_1ru", 0.99, 0.01),
    ("sensitivity_100du_0ru", 1.00, 0.00),
    ("sensitivity_95du_5ru", 0.95, 0.05),
)
REQUIRED_INCIDENCE_SHARE_PAIRS = frozenset({(0.99, 0.01), (1.00, 0.00), (0.95, 0.05)})

ALLOWED_NET_INTEREST_COMPONENT_KEYS = frozenset(
    {
        "fixed_coupon_accrual",
        "bill_discount_amortization",
        "frn_accrual",
        "signed_tips_principal_indexation",
        "public_nonmarketable_interest",
        "offsetting_interest_receipts",
    }
)


def build_fiscal_incidence_policy_rows(
    *,
    scenario_id: str,
    signed_net_primary_flow_bil: float,
    splits: Iterable[tuple[str, float, float]] = DEFAULT_INCIDENCE_SPLITS,
) -> list[dict[str, Any]]:
    """Build central and required sensitivity policy rows."""

    rows: list[dict[str, Any]] = []
    for suffix, du_share, ru_share in splits:
        _validate_share_total(du_share=du_share, ru_share=ru_share, foreign_share=0.0, other_share=0.0)
        rows.append(
            {
                "schema_version": FISCAL_INCIDENCE_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "policy_id": f"{scenario_id}_{suffix}",
                "policy_mode": EXPLICIT_SCENARIO_ASSUMPTION,
                "incidence_basis": SIGNED_NET_PRIMARY_PROXY,
                "signed_net_primary_flow_bil": float(signed_net_primary_flow_bil),
                "primary_outlays_bil": None,
                "primary_receipts_bil": None,
                "recipient_share_source": "explicit_cbo_baseline_scenario_assumption",
                "du_share": float(du_share),
                "ru_share": float(ru_share),
                "foreign_share": 0.0,
                "other_share": 0.0,
                "evidence_status": "not_source_backed_signed_net_proxy",
                "source_role": "scenario_assumption",
                "runtime_role": "memo_only",
                "source_status": "explicit_required_cbo_fiscal_incidence_policy",
                "claim_boundary": "recipient_incidence_assumption_does_not_change_aggregate_signed_flow",
            }
        )
    validate_fiscal_incidence_policy_rows(rows)
    return rows


def validate_fiscal_incidence_policy_rows(rows: Iterable[Mapping[str, Any]]) -> None:
    row_list = list(rows)
    if not row_list:
        raise ValueError("fiscal incidence policy is required")
    for row in row_list:
        if row.get("policy_mode") != EXPLICIT_SCENARIO_ASSUMPTION:
            raise ValueError("first-tranche fiscal incidence rows must be explicit scenario assumptions")
        if row.get("incidence_basis") != SIGNED_NET_PRIMARY_PROXY:
            raise ValueError("first-tranche fiscal incidence rows must use signed_net_primary_proxy")
        if _is_not_blank(row.get("primary_outlays_bil")) or _is_not_blank(row.get("primary_receipts_bil")):
            raise ValueError("gross fields must be blank in signed-net-primary proxy mode")
        _validate_share_total(
            du_share=float(row["du_share"]),
            ru_share=float(row["ru_share"]),
            foreign_share=float(row.get("foreign_share") or 0.0),
            other_share=float(row.get("other_share") or 0.0),
        )


def allocate_signed_primary_flow(row: Mapping[str, Any]) -> dict[str, float]:
    """Allocate signed flow across recipients without changing the aggregate."""

    signed_flow = float(row["signed_net_primary_flow_bil"])
    allocations = {
        "du_flow_bil": signed_flow * float(row["du_share"]),
        "ru_flow_bil": signed_flow * float(row["ru_share"]),
        "foreign_flow_bil": signed_flow * float(row.get("foreign_share") or 0.0),
        "other_flow_bil": signed_flow * float(row.get("other_share") or 0.0),
    }
    allocated_total = sum(allocations.values())
    if abs(allocated_total - signed_flow) > 1e-9:
        raise ValueError("incidence allocation changed aggregate signed primary flow")
    allocations["aggregate_signed_primary_flow_bil"] = signed_flow
    return allocations


def build_fiscal_incidence_sensitivity_results(
    policy_rows: Iterable[Mapping[str, Any]],
    *,
    aggregate_budget_flow_bil: float | None = None,
    aggregate_tga_flow_bil: float | None = None,
) -> list[dict[str, Any]]:
    """Run pure-policy incidence sensitivities and prove aggregate invariance."""

    rows = list(policy_rows)
    validate_fiscal_incidence_policy_rows(rows)
    _validate_required_incidence_sensitivities(rows)
    baseline_signed_flow = float(rows[0]["signed_net_primary_flow_bil"])
    budget_total = baseline_signed_flow if aggregate_budget_flow_bil is None else float(aggregate_budget_flow_bil)
    tga_total = baseline_signed_flow if aggregate_tga_flow_bil is None else float(aggregate_tga_flow_bil)

    results: list[dict[str, Any]] = []
    for row in rows:
        allocation = allocate_signed_primary_flow(row)
        allocated_total = (
            allocation["du_flow_bil"]
            + allocation["ru_flow_bil"]
            + allocation["foreign_flow_bil"]
            + allocation["other_flow_bil"]
        )
        signed_flow = allocation["aggregate_signed_primary_flow_bil"]
        signed_flow_invariant = abs(signed_flow - baseline_signed_flow) <= 1e-9
        budget_total_invariant = abs(budget_total - baseline_signed_flow) <= 1e-9
        tga_total_invariant = abs(tga_total - baseline_signed_flow) <= 1e-9
        allocation_invariant = abs(allocated_total - signed_flow) <= 1e-9
        results.append(
            {
                "schema_version": FISCAL_INCIDENCE_SCHEMA_VERSION,
                "scenario_id": row.get("scenario_id"),
                "policy_id": row.get("policy_id"),
                "du_share": float(row["du_share"]),
                "ru_share": float(row["ru_share"]),
                "foreign_share": float(row.get("foreign_share") or 0.0),
                "other_share": float(row.get("other_share") or 0.0),
                "du_flow_bil": allocation["du_flow_bil"],
                "ru_flow_bil": allocation["ru_flow_bil"],
                "foreign_flow_bil": allocation["foreign_flow_bil"],
                "other_flow_bil": allocation["other_flow_bil"],
                "allocated_total_bil": allocated_total,
                "aggregate_signed_primary_flow_bil": signed_flow,
                "aggregate_budget_flow_bil": budget_total,
                "aggregate_tga_flow_bil": tga_total,
                "aggregate_signed_primary_flow_invariant": signed_flow_invariant,
                "allocation_total_invariant": allocation_invariant,
                "budget_total_invariant": budget_total_invariant,
                "tga_total_invariant": tga_total_invariant,
                "result_status": (
                    "aggregate_invariant"
                    if signed_flow_invariant
                    and allocation_invariant
                    and budget_total_invariant
                    and tga_total_invariant
                    else "aggregate_invariance_failed"
                ),
                "runtime_role": "memo_only",
                "claim_boundary": "incidence_sensitivity_changes_recipient_routing_not_aggregate_budget_or_tga_flow",
            }
        )
    if any(result["result_status"] != "aggregate_invariant" for result in results):
        raise ValueError("incidence sensitivity changed aggregate signed primary flow")
    return results


def build_net_interest_bridge_rows(
    *,
    scenario_id: str,
    fiscal_year: int,
    components: Iterable[Mapping[str, Any]],
    source_vintage: str,
    cbo_reported_net_interest_bil: float | None = None,
) -> list[dict[str, Any]]:
    """Build explicit net-interest bridge metadata rows with no residual plug."""

    rows: list[dict[str, Any]] = []
    for component in components:
        component_key = str(component["component_key"])
        _validate_component_key(component_key)
        rows.append(
            {
                "schema_version": NET_INTEREST_BRIDGE_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "fiscal_year": int(fiscal_year),
                "component_key": component_key,
                "amount_bil": float(component["amount_bil"]),
                "sign_convention": str(component.get("sign_convention", "expense_positive")),
                "include_in_budget_interest": bool(component.get("include_in_budget_interest", True)),
                "component_scope_status": str(component.get("component_scope_status", "explicit_component")),
                "source_family": str(component.get("source_family", "tdcsim_budget_interest_component")),
                "source_table": str(component.get("source_table", "")),
                "source_row_selector": str(component.get("source_row_selector", component_key)),
                "source_vintage": source_vintage,
                "source_role": "diagnostic",
                "runtime_role": "check_only",
                "source_status": str(component.get("source_status", "explicit_component_no_opaque_plug")),
                "claim_boundary": "net_interest_bridge_metadata_cbo_net_interest_nonbinding_check",
            }
        )
    if cbo_reported_net_interest_bil is not None:
        rows.append(
            {
                "schema_version": NET_INTEREST_BRIDGE_SCHEMA_VERSION,
                "scenario_id": scenario_id,
                "fiscal_year": int(fiscal_year),
                "component_key": "cbo_reported_net_interest_check",
                "amount_bil": float(cbo_reported_net_interest_bil),
                "sign_convention": "reported_cbo_positive_expense",
                "include_in_budget_interest": False,
                "component_scope_status": "nonbinding_metadata_check",
                "source_family": "cbo_budget_workbook",
                "source_table": "Table 1-1",
                "source_row_selector": "Net interest",
                "source_vintage": source_vintage,
                "source_role": "diagnostic",
                "runtime_role": "memo_only",
                "source_status": "cbo_reported_net_interest_nonbinding_metadata_check",
                "claim_boundary": "cbo_reported_net_interest_not_a_binding_cash_or_issuance_plug",
            }
        )
    if not rows:
        raise ValueError("net interest bridge rows require at least one explicit component")
    return rows


def _validate_share_total(
    *,
    du_share: float,
    ru_share: float,
    foreign_share: float,
    other_share: float,
) -> None:
    total = du_share + ru_share + foreign_share + other_share
    if abs(total - 1.0) > 1e-12:
        raise ValueError(f"recipient shares must sum to 1.0; found {total:.12f}")


def _validate_component_key(component_key: str) -> None:
    lowered = component_key.lower()
    if "plug" in lowered or "residual" in lowered:
        raise ValueError("net-interest bridge cannot use an opaque plug or residual component")
    if component_key not in ALLOWED_NET_INTEREST_COMPONENT_KEYS:
        raise ValueError(f"unknown net-interest component family: {component_key}")


def _is_not_blank(value: Any) -> bool:
    return value is not None and str(value) != ""


def _validate_required_incidence_sensitivities(rows: Iterable[Mapping[str, Any]]) -> None:
    present = frozenset(
        (
            round(float(row["du_share"]), 12),
            round(float(row["ru_share"]), 12),
        )
        for row in rows
    )
    missing = sorted(REQUIRED_INCIDENCE_SHARE_PAIRS.difference(present))
    if missing:
        formatted = ", ".join(f"{du:.2f}/{ru:.2f}" for du, ru in missing)
        raise ValueError(f"required incidence sensitivities missing: {formatted}")


__all__ = [
    "ALLOWED_NET_INTEREST_COMPONENT_KEYS",
    "build_fiscal_incidence_policy_rows",
    "build_fiscal_incidence_sensitivity_results",
    "build_net_interest_bridge_rows",
    "allocate_signed_primary_flow",
    "validate_fiscal_incidence_policy_rows",
]
