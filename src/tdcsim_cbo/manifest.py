"""Manifest builders for TDCSIM CBO scenario runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .compiler import CboCompiledScenario


def build_run_manifest(
    *,
    scenario_id: str,
    scenario_sha256: str,
    compiled: CboCompiledScenario,
    compiled_manifest_relpath: str,
    start_date: str,
    end_date: str,
    outputs: Mapping[str, Any],
    output_hashes: list[dict[str, Any]],
    boundary_checks: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the public run manifest with explicit unsupported components."""

    return {
        "schema_version": "tdcsim_cbo_run_manifest_v1",
        "scenario_id": scenario_id,
        "scenario_sha256": scenario_sha256,
        "compiled_manifest": compiled_manifest_relpath,
        "compiled_inputs_digest": compiled.compiled_inputs_digest,
        "simulation": {"start_date": start_date, "end_date": end_date, "frequency": "daily"},
        "outputs": dict(outputs),
        "output_hashes": output_hashes,
        "boundary_checks": dict(boundary_checks),
        "unsupported_components": {
            "fed_remittances": "unsupported_in_cbo_scenario_lane",
            "fed_deferred_asset": "unsupported_in_cbo_scenario_lane",
            "complete_budgetary_net_interest_mapping": "unsupported_partial_diagnostic_only",
        },
    }


__all__ = ["build_run_manifest"]
