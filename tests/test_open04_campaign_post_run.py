from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from evaluated_nominal_curve import OPEN04_FIXED_COUPLING, OPEN04_OUTPUT_CONTRACT
from tdcsim_cbo._json import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json,
)
from tdcsim_cbo.bounded_output import (
    ANNUAL_COLUMNS,
    EVENT_SCHEMA_VERSION,
    BoundedResourceLimits,
)
import tdcsim_cbo.open04_campaign as campaign_module
from tdcsim_cbo.open04_campaign import (
    OPEN04_AGGREGATION_CLOCK_ID,
    OPEN04_CAMPAIGN_CONTRACT_ID,
    OPEN04_CAMPAIGN_ROLES,
    OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION,
    OPEN04_END_DATE,
    OPEN04_FREQUENCY,
    OPEN04_START_DATE,
    Open04CampaignError,
    verify_open04_campaign_post_run,
)
import tdcsim_cbo.verifier as verifier_module

from test_open04_campaign_export import (
    _annual_rows,
    _write_gzip_csv,
    open04_pre_run_fixture,
)


_TERMINAL = {
    "baseline": (100.0, 0.0),
    "candidate_a": (99.0, 1.0),
    "candidate_b": (101.0, -1.0),
}
_ROLE_INDEX = {
    "baseline": 0,
    "candidate_a": 1,
    "candidate_b": 2,
}


@pytest.fixture
def open04_post_run_campaign(
    open04_pre_run_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    root = tmp_path / "campaign"
    shutil.copytree(open04_pre_run_fixture["campaign_root"], root)
    contract = deepcopy(open04_pre_run_fixture["contract"])

    common = contract["common_identity"]
    dependency_locks = [
        {
            "relative_path": "uv.lock",
            "sha256": common["uv_lock_sha256"],
            "bytes": 101,
        },
        {
            "relative_path": "requirements.lock.txt",
            "sha256": common["requirements_lock_sha256"],
            "bytes": 202,
        },
    ]
    common["dependency_lock_set_sha256"] = canonical_json_sha256(
        dependency_locks
    )
    contract_path = root / "open04_campaign_contract.json"
    _write_deterministic_json(contract_path, contract)
    contract_sha256 = sha256_file(contract_path)

    run_roots: dict[str, Path] = {}
    for role in OPEN04_CAMPAIGN_ROLES:
        run_root = root / contract["roles"][role]["run_relative_path"]
        run_roots[role] = run_root
        _materialize_complete_run(
            role,
            run_root,
            root=root,
            contract=contract,
            contract_sha256=contract_sha256,
            source_compiled_dir=open04_pre_run_fixture["compiled_dirs"][
                role
            ],
            dependency_locks=dependency_locks,
        )

    def fake_verify_scenario_run(
        run_root: str | Path,
        *,
        baseline_package: str | Path,
        attestation: str | Path,
    ) -> dict[str, Any]:
        del baseline_package, attestation
        manifest = read_json(
            Path(run_root) / "tdcsim_cbo_run_manifest.json"
        )
        bounded = manifest["bounded_evidence"]
        return {
            "status": "pass",
            "verification_grade": "bounded_replay_v1",
            "recomputed": {
                "bounded_event_count": bounded["event_count"],
                "bounded_peak_rss_bytes": bounded["peak_rss_bytes"],
            },
        }

    monkeypatch.setattr(
        verifier_module,
        "verify_scenario_run",
        fake_verify_scenario_run,
    )
    monkeypatch.setattr(
        verifier_module,
        "_verify_code_environment",
        lambda *_args, **_kwargs: None,
    )
    return {
        "root": root,
        "contract": contract,
        "contract_sha256": contract_sha256,
        "run_roots": run_roots,
    }


def test_post_run_verifier_accepts_complete_three_path_campaign(
    open04_post_run_campaign: dict[str, Any],
) -> None:
    fixture = open04_post_run_campaign
    receipt = _verify(fixture)

    assert receipt["status"] == "pass"
    assert receipt["campaign_eligible"] is True
    assert receipt["promotion_status"] == "eligible"
    assert receipt["verification_order"] == list(OPEN04_CAMPAIGN_ROLES)
    assert receipt["pair_gates"]["terminal_sign_gate_status"] == "pass"
    assert receipt["pair_gates"]["overall_wam_ordering"]["status"] == "pass"
    assert receipt["pair_gates"]["full_fy_wam_gate_status"] == "pass"
    assert receipt["pair_gates"]["controller_output_isolation_status"] == "pass"
    assert receipt["pair_gates"]["aggregate_memory_budget_status"] == "pass"
    assert (
        receipt["pair_gates"]["evaluated_delta_antisymmetry_status"]
        == "pass"
    )
    assert {
        role: receipt["roles"][role]["event_schema_version"]
        for role in OPEN04_CAMPAIGN_ROLES
    } == {role: EVENT_SCHEMA_VERSION for role in OPEN04_CAMPAIGN_ROLES}
    assert all(
        receipt["roles"][role]["run_id"] == f"{role}-run-id"
        for role in OPEN04_CAMPAIGN_ROLES
    )
    for role in OPEN04_CAMPAIGN_ROLES:
        summary = (
            fixture["run_roots"][role] / "outputs" / "summary.json"
        )
        assert (
            receipt["roles"][role]["terminal_summary_sha256"]
            == sha256_file(summary)
        )


@pytest.mark.parametrize(
    "mutation",
    ("summary_file_bytes", "summary_manifest_hash", "summary_manifest_bytes"),
)
def test_post_run_verifier_rejects_terminal_summary_mutations(
    open04_post_run_campaign: dict[str, Any],
    mutation: str,
) -> None:
    fixture = open04_post_run_campaign
    run_root = fixture["run_roots"]["candidate_a"]
    summary = run_root / "outputs" / "summary.json"
    manifest_path = run_root / "tdcsim_cbo_run_manifest.json"
    if mutation == "summary_file_bytes":
        summary.write_bytes(summary.read_bytes() + b"\n")
    else:
        manifest = read_json(manifest_path)
        field = "sha256" if mutation == "summary_manifest_hash" else "bytes"
        manifest["output_manifest"]["summary"][field] = (
            "f" * 64
            if field == "sha256"
            else manifest["output_manifest"]["summary"]["bytes"] + 1
        )
        write_json(manifest_path, manifest)

    with pytest.raises(
        Open04CampaignError,
        match="terminal summary differs from its output manifest",
    ):
        _verify(fixture)


def test_post_run_verifier_rejects_controller_terminal_digest(
    open04_post_run_campaign: dict[str, Any],
) -> None:
    fixture = open04_post_run_campaign
    receipt_path = _controller_receipt_path(fixture, "candidate_b")
    receipt = read_json(receipt_path)
    receipt["terminal_summary_sha256"] = "f" * 64
    write_json(receipt_path, receipt)

    with pytest.raises(
        Open04CampaignError,
        match="completion receipt identity/status differs",
    ):
        _verify(fixture)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("controller_split", "share one batch controller"),
        ("controller_memory", "aggregate memory acceptance budget"),
        ("controller_process_live", "identity/status differs"),
        ("event_schema", "run event schema differs"),
        ("blank_run_id", "run_id is blank"),
        ("curve_short_end", "run curve receipt differs"),
        ("accounting", "annual TDC overlap identity failed"),
        ("terminal_sign", "failed_predeclared_paired_sign_gate"),
        ("full_fy_wam", "FY2027 realized WAM ordering failed"),
    ),
)
def test_post_run_verifier_rejects_campaign_gate_mutations(
    open04_post_run_campaign: dict[str, Any],
    mutation: str,
    match: str,
) -> None:
    fixture = open04_post_run_campaign
    if mutation in {
        "controller_split",
        "controller_memory",
        "controller_process_live",
    }:
        receipt_path = _controller_receipt_path(fixture, "candidate_a")
        receipt = read_json(receipt_path)
        if mutation == "controller_split":
            receipt["controller_run_id"] = "controller-other-batch"
            receipt["controller_telemetry_locator"] = (
                "controller_run_id:controller-other-batch"
            )
        elif mutation == "controller_memory":
            receipt["controller_peak_rss_mb"] = 11 * 1024
        else:
            receipt["controller_process_tree_drained"] = False
        write_json(receipt_path, receipt)
    elif mutation in {"event_schema", "blank_run_id", "curve_short_end"}:
        role = "candidate_a"
        manifest_path = (
            fixture["run_roots"][role]
            / "tdcsim_cbo_run_manifest.json"
        )
        manifest = read_json(manifest_path)
        if mutation == "event_schema":
            manifest["bounded_evidence"]["event_schema_version"] = (
                "tdcsim_event_commitment_v1"
            )
        elif mutation == "blank_run_id":
            manifest["run_id"] = ""
        else:
            manifest["evaluated_nominal_curve"][
                "short_end_bitwise_mismatch_count"
            ] = 1
        write_json(manifest_path, manifest)
    elif mutation == "accounting":
        rows = _read_annual_rows(fixture, "candidate_a")
        rows[-1]["tdc_change_ex_overlap_bil"] = "7.0"
        _write_annual_rows(fixture, "candidate_a", rows)
    elif mutation == "terminal_sign":
        _replace_annual_rows(
            fixture,
            "candidate_a",
            issuance_role="candidate_a",
            terminal_cost=101.0,
            terminal_tdc=1.0,
        )
    elif mutation == "full_fy_wam":
        _replace_annual_rows(
            fixture,
            "candidate_a",
            issuance_role="baseline",
            terminal_cost=99.0,
            terminal_tdc=1.0,
        )
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(mutation)

    with pytest.raises(Open04CampaignError, match=match):
        _verify(fixture)


def _verify(fixture: dict[str, Any]) -> dict[str, Any]:
    return verify_open04_campaign_post_run(
        fixture["root"],
        expected_contract_sha256=fixture["contract_sha256"],
        baseline_package=Path("synthetic-baseline.zip"),
        attestation=Path("synthetic-attestation.json"),
    )


def _materialize_complete_run(
    role: str,
    run_root: Path,
    *,
    root: Path,
    contract: dict[str, Any],
    contract_sha256: str,
    source_compiled_dir: Path,
    dependency_locks: list[dict[str, Any]],
) -> None:
    shutil.copytree(source_compiled_dir.parent, run_root / "compile")
    source_scenario = (
        root / contract["roles"][role]["scenario_source_relative_path"]
    )
    scenario_path = run_root / "scenario.json"
    shutil.copyfile(source_scenario, scenario_path)

    outputs = run_root / "outputs"
    outputs.mkdir()
    summary_path = outputs / "summary.json"
    write_json(
        summary_path,
        {
            "status": "complete",
            "run_id": f"{role}-run-id",
            "scenario_role": role,
        },
    )

    issuance_mix = read_json(
        run_root
        / "compile"
        / "compiled"
        / "forecast_inputs"
        / "tdcsim_issuance_mix_assumptions.json"
    )
    terminal_cost, terminal_tdc = _TERMINAL[role]
    annual_path = outputs / "tdcsim_annual_economic_summary.csv.gz"
    _write_gzip_csv(
        annual_path,
        ANNUAL_COLUMNS,
        _annual_rows(
            role,
            issuance_mix,
            terminal_cost=terminal_cost,
            terminal_tdc=terminal_tdc,
        ),
    )

    declared = contract["roles"][role]
    common = contract["common_identity"]
    output_manifest = {
        "profile": OPEN04_OUTPUT_CONTRACT["profile"],
        "compression": OPEN04_OUTPUT_CONTRACT["compression"],
        "summary": {
            "path": "summary.json",
            "sha256": sha256_file(summary_path),
            "bytes": summary_path.stat().st_size,
        },
    }
    limits = BoundedResourceLimits()
    peak_rss_bytes = 512 * 1024**2
    event_count = 100 + _ROLE_INDEX[role]
    manifest: dict[str, Any] = {
        "schema_version": "tdcsim_cbo_scenario_run_manifest_v2",
        "status": "complete",
        "verification_grade": "bounded_replay_v1",
        "evidence_profile": "bounded_period_closure_v1",
        "run_id": f"{role}-run-id",
        "open04_campaign": read_json(source_scenario)["open04_campaign"],
        "baseline": {
            "package_id": common["package_id"],
            "package_sha256": common["baseline_package_sha256"],
            "manifest_sha256": common["baseline_manifest_sha256"],
            "release_attestation_sha256": common[
                "release_attestation_sha256"
            ],
        },
        "simulation": {
            "start_date": OPEN04_START_DATE,
            "end_date": OPEN04_END_DATE,
            "frequency": OPEN04_FREQUENCY,
        },
        "aggregation_clock": {
            "clock_id": OPEN04_AGGREGATION_CLOCK_ID,
        },
        "output_manifest": output_manifest,
        "coupling_decisions": dict(OPEN04_FIXED_COUPLING),
        "scenario": {
            "scenario_id": declared["scenario_id"],
            "canonical_sha256": declared["scenario_sha256"],
            "relative_path": "scenario.json",
            "source_file_sha256": sha256_file(scenario_path),
        },
        "execution_contract": {
            "schema_version": "tdcsim_cbo_execution_contract_v1",
            "single_writer_claim": True,
            "writer_claim_file_name": ".tdcsim-cbo-bounded-writer.claim",
            "writer_claim_scope": "output_parent",
            "writer_claim_scope_id": f"{OPEN04_CAMPAIGN_CONTRACT_ID}.{role}",
            "one_scenario_per_worker": True,
            "process_pool_enabled": False,
            "parent_watchdog_required": True,
            "scenario_process_mode": "parent_watchdog_worker",
            "numerical_thread_environment": {
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
            },
        },
        "bounded_evidence": {
            "memory_thresholds": {
                "minimum_available_bytes": limits.minimum_available_bytes,
                "acceptance_peak_rss_bytes": (
                    limits.acceptance_peak_rss_bytes
                ),
                "application_abort_rss_bytes": (
                    limits.application_abort_rss_bytes
                ),
                "parent_graceful_stop_rss_bytes": (
                    limits.parent_graceful_stop_rss_bytes
                ),
                "parent_kill_rss_bytes": limits.parent_kill_rss_bytes,
            },
            "peak_rss_bytes": peak_rss_bytes,
            "invariant_status": "pass",
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_count": event_count,
            "event_root_sha256": _digest(f"{role}-events"),
        },
        "parent_watchdog": {
            "status": "accepted",
            "action": "completed",
            "child_returncode": 0,
        },
        "compiled_manifest": (
            "compile/compiled/tdcsim_cbo_compiled_manifest.json"
        ),
        "code_environment": _code_environment(
            common,
            dependency_locks=dependency_locks,
        ),
    }
    if role != "baseline":
        manifest["evaluated_nominal_curve"] = {
            "sidecar_sha256": declared["curve_sidecar_sha256"],
            "evaluated_delta_sha256": declared[
                "compiled_curve_delta_digest"
            ],
            "short_end_bitwise_mismatch_count": 0,
            "max_abs_analytic_delta_error_decimal": 0.0,
        }
    manifest_path = run_root / "tdcsim_cbo_run_manifest.json"
    write_json(manifest_path, manifest)

    controller_run_id = "controller-open04-batch"
    controller_receipt = {
        "schema_version": OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION,
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": contract_sha256,
        "role": role,
        "run_id": manifest["run_id"],
        "host": "BZOT",
        "controller": "remote_controller",
        "placement": "auto",
        "terminal_status": "completed",
        "started_at_utc": "2026-07-30T00:00:00Z",
        "completed_at_utc": "2026-07-30T02:30:00Z",
        "worker_exit_confirmed_at_utc": "2026-07-30T02:31:00Z",
        "terminal_summary_sha256": sha256_file(summary_path),
        "run_manifest_sha256": sha256_file(manifest_path),
        "controller_run_id": controller_run_id,
        "controller_summary_sha256": _digest("batch-controller-summary"),
        "controller_command_sha256": _digest("batch-controller-command"),
        "controller_command_exit_code": 0,
        "controller_exit_code": 0,
        "preflight_conflicts": 0,
        "postrun_conflicts": 0,
        "controller_peak_rss_mb": 512.0,
        "controller_avg_cpu_pct": 50.0,
        "controller_memory_guard_status": "pass",
        "controller_telemetry_status": "reported",
        "controller_telemetry_locator": (
            f"controller_run_id:{controller_run_id}"
        ),
        "controller_process_tree_drained": True,
    }
    write_json(
        root
        / declared["controller_completion_receipt_relative_path"],
        controller_receipt,
    )


def _code_environment(
    common: dict[str, Any],
    *,
    dependency_locks: list[dict[str, Any]],
) -> dict[str, Any]:
    environment = {
        key: common[key] for key in campaign_module._CODE_IDENTITY_KEYS
    }
    environment["wheel_artifact"] = {
        "sha256": common["wheel_artifact_sha256"],
    }
    environment["producer_source_identity"] = {
        "source_tree": {
            "dependency_lock_files": dependency_locks,
            "dependency_lock_set_sha256": common[
                "dependency_lock_set_sha256"
            ],
            "release_commit_sha": common["code_commit_sha"],
        },
        "installed_archive_sha256": common["wheel_sha256"],
    }
    return environment


def _controller_receipt_path(
    fixture: dict[str, Any],
    role: str,
) -> Path:
    return (
        fixture["root"]
        / fixture["contract"]["roles"][role][
            "controller_completion_receipt_relative_path"
        ]
    )


def _read_annual_rows(
    fixture: dict[str, Any],
    role: str,
) -> list[dict[str, str]]:
    import csv
    import gzip

    path = (
        fixture["run_roots"][role]
        / "outputs"
        / "tdcsim_annual_economic_summary.csv.gz"
    )
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_annual_rows(
    fixture: dict[str, Any],
    role: str,
    rows: list[dict[str, Any]],
) -> None:
    path = (
        fixture["run_roots"][role]
        / "outputs"
        / "tdcsim_annual_economic_summary.csv.gz"
    )
    _write_gzip_csv(path, ANNUAL_COLUMNS, rows)


def _replace_annual_rows(
    fixture: dict[str, Any],
    role: str,
    *,
    issuance_role: str,
    terminal_cost: float,
    terminal_tdc: float,
) -> None:
    issuance_mix = read_json(
        fixture["run_roots"][issuance_role]
        / "compile"
        / "compiled"
        / "forecast_inputs"
        / "tdcsim_issuance_mix_assumptions.json"
    )
    _write_annual_rows(
        fixture,
        role,
        _annual_rows(
            role,
            issuance_mix,
            terminal_cost=terminal_cost,
            terminal_tdc=terminal_tdc,
        ),
    )


def _write_deterministic_json(path: Path, value: Any) -> None:
    path.write_bytes(
        (
            json.dumps(
                value,
                sort_keys=True,
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    )


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
