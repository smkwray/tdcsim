"""Manifest builders for TDCSIM CBO scenario runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .baseline import CboBaselinePackage
from .compiler import CboCompiledScenario
from .contract import CboScenarioSpec


def build_run_manifest(
    *,
    scenario_id: str,
    scenario_sha256: str,
    compiled: CboCompiledScenario,
    compiled_manifest_relpath: str,
    baseline: CboBaselinePackage,
    scenario: CboScenarioSpec,
    scenario_relpath: str,
    scenario_file_sha256: str,
    scenario_referenced_files: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    outputs: Mapping[str, Any],
    output_hashes: list[dict[str, Any]],
    boundary_checks: Mapping[str, Any],
    code_environment: Mapping[str, Any],
    generated_at_utc: str,
) -> dict[str, Any]:
    """Build the public run manifest with explicit unsupported components."""

    validation = _validation(boundary_checks)
    return {
        "schema_version": "tdcsim_cbo_scenario_run_manifest_v1",
        "run_id": f"{scenario_id}-{scenario_sha256[:12]}",
        "status": "complete",
        "verification_grade": "local",
        "generated_at_utc": generated_at_utc,
        "baseline": {
            "package_id": baseline.package_id,
            "package_sha256": baseline.package_sha256,
            "manifest_sha256": baseline.manifest_sha256,
            "release_attestation_sha256": baseline.attestation.sha256,
        },
        "scenario": {
            "scenario_id": scenario_id,
            "canonical_sha256": scenario_sha256,
            "source_file_sha256": scenario_file_sha256,
            "relative_path": scenario_relpath,
            "referenced_file_hashes": _referenced_file_hashes(scenario),
            "referenced_files": scenario_referenced_files,
        },
        "code_environment": dict(code_environment),
        "claim_boundary": {
            "cash_residual_role": "reconciliation_only_not_issuance_sizing",
            "net_interest_role": "diagnostic_nonbinding",
            "fed_holdings_role": "holder_stock_target_not_fed_auction_purchase",
            "fed_remittance_deferred_asset": "unsupported_in_cbo_scenario_lane",
            "complete_budgetary_net_interest_mapping": "unsupported_partial_diagnostic_only",
        },
        "coupling_decisions": dict(scenario.data.get("coupling", {})),
        "compiled_manifest": compiled_manifest_relpath,
        "compiled_inputs_digest": compiled.compiled_inputs_digest,
        "compiled_inputs": [
            {
                "logical_name": item["path"],
                "relative_path": f"compile/compiled/forecast_inputs/{item['path']}",
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "media_type": _media_type(item["path"]),
            }
            for item in compiled.manifest.get("input_hashes", [])
        ],
        "simulation": {"start_date": start_date, "end_date": end_date, "frequency": "daily"},
        "outputs": [
            {
                "logical_name": item["path"],
                "relative_path": f"outputs/{item['path']}",
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "media_type": _media_type(item["path"]),
            }
            for item in output_hashes
        ],
        "output_manifest": dict(outputs),
        "output_hashes": output_hashes,
        "boundary_checks": dict(boundary_checks),
        "validation": validation,
        "unsupported_components": [
            "fed_remittances",
            "fed_deferred_asset",
            "complete_budgetary_net_interest_mapping",
        ],
    }


def _validation(boundary_checks: Mapping[str, Any]) -> dict[str, Any]:
    gates = [
        {
            "id": "fed_target_holder_allocation_only",
            "status": "pass" if boundary_checks.get("fed_target_holder_allocation_only") is True else "fail",
            "observed": str(boundary_checks.get("max_abs_fed_auction_face", "")),
        },
        {
            "id": "net_interest_diagnostic_nonbinding",
            "status": "pass" if boundary_checks.get("net_interest_role") == "diagnostic_nonbinding" else "fail",
        },
        {
            "id": "remittance_deferred_asset_unsupported",
            "status": "pass"
            if boundary_checks.get("remittance_deferred_asset_status") == "unsupported_in_cbo_scenario_lane"
            else "fail",
        },
    ]
    return {
        "status": "pass" if all(item["status"] == "pass" for item in gates) else "fail",
        "gates": gates,
        "invariants": [
            {
                "id": "cash_residual_not_issuance_sizing",
                "status": "pass"
                if "False" in boundary_checks.get("cash_residual_affects_issuance_size", [])
                or boundary_checks.get("cash_residual_affects_issuance_size", []) == []
                else "fail",
            }
        ],
    }


def _referenced_file_hashes(scenario: CboScenarioSpec) -> dict[str, str]:
    out: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            if "relative_path" in value and "sha256" in value:
                out[str(value["relative_path"])] = str(value["sha256"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(scenario.data)
    return out


def _media_type(path: str) -> str:
    if path.endswith(".json"):
        return "application/json"
    if path.endswith(".csv") or path.endswith(".csv.gz"):
        return "text/csv"
    if path.endswith(".sqlite"):
        return "application/vnd.sqlite3"
    return "application/octet-stream"


__all__ = ["build_run_manifest"]
