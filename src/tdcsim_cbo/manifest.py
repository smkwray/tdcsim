"""Manifest builders for TDCSIM CBO scenario runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .baseline import CboBaselinePackage
from .compiler import CboCompiledScenario
from .contract import CboScenarioSpec


RUN_CLAIM_BOUNDARY = {
    "cash_residual_role": "reconciliation_only_not_issuance_sizing",
    "net_interest_role": "diagnostic_nonbinding",
    "fed_holdings_role": "holder_stock_target_not_fed_auction_purchase",
    "fed_remittance_deferred_asset": "unsupported_in_cbo_scenario_lane",
    "complete_budgetary_net_interest_mapping": "unsupported_partial_diagnostic_only",
}

RUN_UNSUPPORTED_COMPONENTS = [
    "fed_remittances",
    "fed_deferred_asset",
    "complete_budgetary_net_interest_mapping",
]


def record_parent_watchdog_acceptance(
    manifest: Mapping[str, Any],
    *,
    child_pid: int,
    child_returncode: int,
    action: str,
    peak_rss_bytes: int,
    worker_peak_rss_bytes: int,
    acceptance_peak_rss_bytes: int,
    terminate_rss_bytes: int,
    kill_rss_bytes: int,
    poll_interval_seconds: float,
) -> dict[str, Any]:
    """Finalize a staged manifest with the parent's full-child RSS observation."""

    if manifest.get("schema_version") != "tdcsim_cbo_scenario_run_manifest_v2":
        raise ValueError("parent watchdog acceptance requires a bounded v2 manifest")
    if manifest.get("status") != "pending_parent_watchdog_acceptance":
        raise ValueError("manifest is not awaiting parent watchdog acceptance")
    peak = int(peak_rss_bytes)
    worker_peak = int(worker_peak_rss_bytes)
    limit = int(acceptance_peak_rss_bytes)
    effective_peak = max(peak, worker_peak)
    if (
        action != "completed"
        or int(child_returncode) != 0
        or peak < 0
        or worker_peak < 0
        or limit <= 0
        or effective_peak > limit
    ):
        raise ValueError(
            "parent watchdog result is not acceptable: "
            f"action={action}, returncode={child_returncode}, "
            f"parent_peak={peak}, worker_peak={worker_peak}, limit={limit}"
        )
    if int(terminate_rss_bytes) <= limit:
        raise ValueError("parent terminate threshold must exceed acceptance threshold")
    if int(kill_rss_bytes) <= int(terminate_rss_bytes):
        raise ValueError("parent kill threshold must exceed terminate threshold")
    interval = float(poll_interval_seconds)
    if interval <= 0.0:
        raise ValueError("parent watchdog poll interval must be positive")

    bounded = manifest.get("bounded_evidence")
    if not isinstance(bounded, Mapping):
        raise ValueError("bounded evidence is missing from staged manifest")
    thresholds = bounded.get("memory_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("bounded memory thresholds are missing from staged manifest")
    expected = {
        "acceptance_peak_rss_bytes": limit,
        "parent_graceful_stop_rss_bytes": int(terminate_rss_bytes),
        "parent_kill_rss_bytes": int(kill_rss_bytes),
    }
    for key, value in expected.items():
        if int(thresholds.get(key, -1)) != value:
            raise ValueError(f"parent watchdog threshold disagrees with {key}")

    finalized = dict(manifest)
    finalized["status"] = "complete"
    finalized["parent_watchdog"] = {
        "status": "accepted",
        "sampler": "parent_process_rss_poll_v1",
        "child_pid": int(child_pid),
        "child_returncode": int(child_returncode),
        "action": action,
        "peak_rss_bytes": peak,
        "worker_peak_rss_bytes": worker_peak,
        "effective_peak_rss_bytes": effective_peak,
        "acceptance_peak_rss_bytes": limit,
        "terminate_rss_bytes": int(terminate_rss_bytes),
        "kill_rss_bytes": int(kill_rss_bytes),
        "poll_interval_seconds": interval,
    }
    validation = dict(finalized.get("validation") or {})
    invariants = list(validation.get("invariants") or [])
    invariants.append(
        {
            "id": "parent_watchdog_peak_rss",
            "status": "pass",
            "observed": effective_peak,
            "limit": limit,
        }
    )
    validation["invariants"] = invariants
    finalized["validation"] = validation
    milestones = list(finalized.get("execution_milestones") or [])
    milestones.append("parent_watchdog_accepted")
    finalized["execution_milestones"] = milestones
    return finalized


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
    fiscal_incidence_policy_id: str,
    outputs: Mapping[str, Any],
    output_hashes: list[dict[str, Any]],
    boundary_checks: Mapping[str, Any],
    code_environment: Mapping[str, Any],
    generated_at_utc: str,
    bounded_evidence: Mapping[str, Any] | None = None,
    evaluated_nominal_curve: Mapping[str, Any] | None = None,
    execution_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public run manifest with explicit unsupported components."""

    validation = validation_from_boundary_checks(boundary_checks)
    bounded = dict(bounded_evidence or {})
    if bounded:
        if bounded.get("evidence_profile") != "bounded_period_closure_v1":
            raise ValueError("unsupported bounded evidence profile")
        if bounded.get("verification_grade") != "bounded_replay_v1":
            raise ValueError("unsupported bounded verification grade")
        validation = {
            **validation,
            "invariants": [
                *validation["invariants"],
                {
                    "id": "bounded_period_closure",
                    "status": "pass",
                    "observed": f"periods={int(bounded.get('period_count', 0))}",
                },
                {
                    "id": "portfolio_row_budget",
                    "status": (
                        "pass"
                        if int(bounded.get("max_portfolio_rows", 0))
                        <= int(bounded.get("portfolio_row_budget", 0))
                        else "fail"
                    ),
                    "observed": int(bounded.get("max_portfolio_rows", 0)),
                    "limit": int(bounded.get("portfolio_row_budget", 0)),
                },
                {
                    "id": "key_cardinality_budget",
                    "status": (
                        "pass"
                        if int(bounded.get("max_key_cardinality", 0))
                        <= int(bounded.get("key_cardinality_budget", 0))
                        else "fail"
                    ),
                    "observed": int(bounded.get("max_key_cardinality", 0)),
                    "limit": int(bounded.get("key_cardinality_budget", 0)),
                },
            ],
        }
    manifest = {
        "schema_version": (
            "tdcsim_cbo_scenario_run_manifest_v2"
            if bounded
            else "tdcsim_cbo_scenario_run_manifest_v1"
        ),
        "run_id": f"{scenario_id}-{scenario_sha256[:12]}",
        "status": "complete",
        "verification_grade": (
            str(bounded["verification_grade"]) if bounded else "local"
        ),
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
        "claim_boundary": dict(RUN_CLAIM_BOUNDARY),
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
        "fiscal_incidence_policy_id": fiscal_incidence_policy_id,
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
        "unsupported_components": list(RUN_UNSUPPORTED_COMPONENTS),
        "execution_contract": dict(execution_contract or {}),
    }
    if bounded:
        manifest.update(
            {
                "evidence_profile": str(bounded["evidence_profile"]),
                "aggregation_clock": {
                    "clock_id": "federal_fiscal_year_period_end_v1",
                    "bucket_label_rule": "federal_fiscal_year_by_period_end",
                    "opening_partial_policy": "unannualized_separate_partial_bucket",
                    "snapshot_date_rule": "exact_september_30",
                },
                "bounded_evidence": {
                    "event_schema_version": str(
                        bounded["event_schema_version"]
                    ),
                    "event_count": int(bounded["event_count"]),
                    "event_root_sha256": str(bounded["event_root_sha256"]),
                    "final_state_sha256": str(bounded["final_state_sha256"]),
                    "period_count": int(bounded["period_count"]),
                    "portfolio_row_budget": int(
                        bounded["portfolio_row_budget"]
                    ),
                    "max_portfolio_rows": int(bounded["max_portfolio_rows"]),
                    "max_active_portfolio_rows": int(
                        bounded["max_active_portfolio_rows"]
                    ),
                    "key_cardinality_budget": int(
                        bounded["key_cardinality_budget"]
                    ),
                    "max_key_cardinality": int(
                        bounded["max_key_cardinality"]
                    ),
                    "peak_rss_bytes": int(bounded["peak_rss_bytes"]),
                    "memory_thresholds": dict(bounded["memory_thresholds"]),
                    "deterministic_artifacts": dict(
                        bounded["deterministic_artifacts"]
                    ),
                    "resource_artifact": dict(
                        bounded["artifacts"]["resources"]
                    ),
                    "invariant_status": "pass",
                },
                "execution_milestones": [
                    "compile_complete",
                    "engine_started",
                    "engine_complete",
                    "source_evidence_finalized",
                    "bounded_period_closure_passed",
                    "run_manifest_finalized",
                ],
            }
        )
    if evaluated_nominal_curve is not None:
        manifest["evaluated_nominal_curve"] = dict(
            evaluated_nominal_curve
        )
    return manifest


def validation_from_boundary_checks(boundary_checks: Mapping[str, Any]) -> dict[str, Any]:
    cash_residual_affects_issuance_size = _normalized_bool_set(
        boundary_checks.get("cash_residual_affects_issuance_size", [])
    )
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
    invariants = [
        {
            "id": "cash_residual_not_issuance_sizing",
            "status": "pass"
            if cash_residual_affects_issuance_size == {False}
            else "fail",
            "observed": ",".join(str(value) for value in boundary_checks.get("cash_residual_affects_issuance_size", [])),
        },
        *cash_closure_validation_invariants(boundary_checks),
    ]
    return {
        "status": "pass"
        if all(item["status"] == "pass" for item in gates)
        and all(item["status"] == "pass" for item in invariants)
        else "fail",
        "gates": gates,
        "invariants": invariants,
    }


def cash_closure_validation_invariants(boundary_checks: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Render the cash-chain validation claims from independently measurable values."""

    return [
        # The modeled Treasury cash chain must actually close. A negative TGA is an
        # unmodeled overdraft; a cash residual booked to the TGA alone creates cash
        # with no counterparty. Either one makes the run's TDC an open-chain figure,
        # so both fail the run rather than being reported as diagnostics.
        {
            "id": "tga_nonnegative",
            "status": "pass" if boundary_checks.get("tga_nonnegative") is True else "fail",
            "observed": (
                f"min_tga={boundary_checks.get('min_tga', '')};"
                f"negative_periods={boundary_checks.get('negative_tga_periods', '')}"
            ),
        },
        {
            "id": "cash_residual_fully_booked",
            "status": "pass" if boundary_checks.get("cash_residual_fully_booked") is True else "fail",
            "observed": str(boundary_checks.get("sum_abs_unbooked_cash_residual", "")),
        },
        # CBO's published "other means of financing" row is the only source-controlled
        # account that could absorb a cash gap, and it closes CBO's own debt identity
        # exactly at a scale of tens of billions per year. The modeled gap exceeds it by
        # roughly two orders of magnitude, so no decomposition of that control could close
        # it and calling the difference OMF would be a project plug wearing a CBO label.
        #
        # Reported, not gated. The operating-cash path is a scenario proxy
        # (source_role=scenario_assumption), so failing on its gap would fail every run
        # against an assumption the project does not hold. Note this is NOT because
        # tga_nonnegative already covers it — a positive but off-target TGA passes that
        # gate while carrying a material gap. What must be visible is the comparison
        # itself, next to the only account that could ever have absorbed it.
        {
            "id": "operating_cash_gap_measured_against_cbo_omf_control",
            "status": "pass",
            "observed": (
                f"target_met={boundary_checks.get('operating_cash_target_met', 'unknown')};"
                f"reconcilable_to_omf={boundary_checks.get('cash_gap_within_omf_control', 'unknown')};"
                f"max_gap_bil={boundary_checks.get('max_abs_operating_cash_gap_bil', '')};"
                f"max_cbo_omf_bil={boundary_checks.get('max_abs_cbo_omf_control_bil', '')};"
                f"targeted_periods={boundary_checks.get('operating_cash_targeted_periods', '')}"
            ),
        },
    ]


def _normalized_bool_set(values: Any) -> set[bool]:
    if isinstance(values, (str, bool)) or values is None:
        iterable = [values]
    elif isinstance(values, list):
        iterable = values
    else:
        return set()
    out: set[bool] = set()
    for value in iterable:
        if isinstance(value, bool):
            out.add(value)
        elif isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "false":
                out.add(False)
            elif lowered == "true":
                out.add(True)
            else:
                return set()
        else:
            return set()
    return out


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


__all__ = [
    "RUN_CLAIM_BOUNDARY",
    "RUN_UNSUPPORTED_COMPONENTS",
    "build_run_manifest",
    "record_parent_watchdog_acceptance",
    "validation_from_boundary_checks",
]
