from __future__ import annotations

import csv
from copy import deepcopy
from datetime import date
import gzip
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import pytest

from evaluated_nominal_curve import OPEN04_FIXED_COUPLING, OPEN04_OUTPUT_CONTRACT
from tdcsim_cbo import CboScenarioSpec
from tdcsim_cbo import CboScenarioCompiler
from tdcsim_cbo._json import (
    canonical_json_sha256,
    read_json,
    sha256_file,
    write_json,
)
from tdcsim_cbo.bounded_output import (
    ANNUAL_COLUMNS,
    EVENT_SCHEMA_VERSION,
    RESOURCE_COLUMNS,
    BoundedResourceLimits,
)
from tdcsim_cbo.open04_campaign import (
    OPEN04_BASELINE_IDENTITY,
    OPEN04_CAMPAIGN_CONTRACT_ID,
    OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION,
    OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION,
    OPEN04_FUNDING_CLOSURE_MODE,
    OPEN04_POST_COMMON_IDENTITY_KEYS,
    OPEN04_POST_RUN_RECEIPT_SCHEMA_VERSION,
    OPEN04_ROLE_TO_ISSUANCE_MIX_OVERRIDE_SHA256,
    OPEN04_ROLE_TO_SCENARIO_ID,
    Open04CampaignError,
    build_open04_scenario_mappings,
    freeze_open04_campaign_contract,
    validate_open04_campaign_post_receipt_mapping,
    verify_open04_campaign_post_run,
    verify_open04_campaign_pre_run,
)
import tdcsim_cbo.open04_campaign as open04_campaign_module
import tdcsim_cbo.open04_export as open04_export_module
from tdcsim_cbo.open04_export import (
    MATURITY_COLUMNS,
    OPEN04_EXPORT_FILES,
    SCENARIO_INPUT_COLUMNS,
    TDC_PATH_COLUMNS,
    Open04ExportError,
    export_open04_thin_package,
)
import tdcsim_cbo.runner as runner_module
import tdcsim_cbo.verifier as verifier_module
from tdcsim_cbo.runner import RunnerError, run_cbo_scenario
from tdcsim_cbo.verifier import (
    VerificationError,
    _verify_bounded_manifest_contract,
    _verify_run_evaluated_nominal_curve,
)
from tdcsim_cbo.process_watchdog import THREAD_LIMIT_ENVIRONMENT_VARIABLES

from test_cbo_bounded_verifier import (
    _add_parent_watchdog_acceptance,
    _package,
)
from test_open04_campaign_baseline import _full_horizon_compiler_baseline


_START = "2026-06-21"
_END = "2036-09-30"
_EXPECTED_SCENARIO_SHA256 = {
    "baseline": "59e2dd1395efbbe18e9d633c49140f642d0a487b0ca0c4e1381dc84dade5946d",
    "candidate_a": "832424470210433e0344d3f5676c8b3be64acb8b7add5602a67abe309c677e86",
    "candidate_b": "321fc0c76942edab370549d6b869d0a455c449c0807b8f9625aa0db0193a4746",
}
_EXPECTED_MIX_SHA256 = {
    "baseline": "none",
    "candidate_a": "ce4109c4dde43233e988a9d46228c26e39ce96f1455b6ca1e3caf66177bae860",
    "candidate_b": "1c2f666215add3e7a03ad88e5c2b7100c4349efb8bab57188ce8e97c024d16e1",
}
_EXPECTED_WAM = {
    "baseline": 6.8675,
    "candidate_a": 5.867502794602249,
    "candidate_b": 7.8675090182959995,
}


def test_three_path_writer_freezes_exact_mix_wam_sign_and_hashes() -> None:
    mixes = _candidate_mixes()
    scenarios = build_open04_scenario_mappings(
        baseline_identity=OPEN04_BASELINE_IDENTITY,
        candidate_a_issuance_mix=mixes["candidate_a"],
        candidate_b_issuance_mix=mixes["candidate_b"],
    )

    assert tuple(scenarios) == ("baseline", "candidate_a", "candidate_b")
    assert dict(OPEN04_ROLE_TO_ISSUANCE_MIX_OVERRIDE_SHA256) == (
        _EXPECTED_MIX_SHA256
    )
    assert {
        role: canonical_json_sha256(scenario)
        for role, scenario in scenarios.items()
    } == _EXPECTED_SCENARIO_SHA256
    assert scenarios["baseline"]["overrides"] == {}
    assert scenarios["candidate_a"]["overrides"]["holder_preferences"] == {
        "mode": "static_shares",
        "rows": [
            {
                "security_type": "bonds",
                "shares": {
                    "Banks": 0.11526315789473685,
                    "CB": 0.0,
                    "FedInternal": 0.0,
                    "Foreign": 0.2631578947368421,
                    "Private": 0.621578947368421,
                    "TrustFunds": 0.0,
                },
            }
        ],
    }
    assert "holder_preferences" not in scenarios["candidate_b"]["overrides"]
    assert {
        scenario["open04_campaign"]["funding_closure_mode"]
        for scenario in scenarios.values()
    } == {"cbo_debt_reference_plus_tga_floor_financing_v1"}

    observed_wam = {
        "baseline": _EXPECTED_WAM["baseline"],
        **{
            role: _issuance_wam_years(
                scenario["overrides"]["issuance_mix"]
            )
            for role, scenario in scenarios.items()
            if role != "baseline"
        },
    }
    assert observed_wam == pytest.approx(_EXPECTED_WAM, abs=1e-12)
    assert (
        observed_wam["candidate_a"]
        < observed_wam["baseline"]
        < observed_wam["candidate_b"]
    )
    assert {
        role: (
            0.0
            if role == "baseline"
            else float(
                scenario["overrides"]["nominal_yield_curve"]["shocks"][1][
                    "shock_bp"
                ]
            )
        )
        for role, scenario in scenarios.items()
    } == {"baseline": 0.0, "candidate_a": -25.0, "candidate_b": 25.0}


@pytest.mark.parametrize(
    ("role", "path", "value"),
    (
        ("candidate_a", ("fixed_remainder_shares", "bills"), 0.335031),
        ("candidate_b", ("maturity_distributions", "notes", 0, "share"), 0.8),
    ),
)
def test_three_path_writer_rejects_mix_retuning(
    role: str,
    path: tuple[Any, ...],
    value: float,
) -> None:
    mixes = _candidate_mixes()
    target: Any = mixes[role]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(
        Open04CampaignError, match="differs from the frozen OPEN-04 input"
    ):
        build_open04_scenario_mappings(
            baseline_identity=OPEN04_BASELINE_IDENTITY,
            candidate_a_issuance_mix=mixes["candidate_a"],
            candidate_b_issuance_mix=mixes["candidate_b"],
        )


def test_frozen_campaign_and_pre_run_receipt_bind_all_three_inputs(
    open04_pre_run_fixture: dict[str, Any],
) -> None:
    contract = open04_pre_run_fixture["contract"]
    receipt = verify_open04_campaign_pre_run(
        open04_pre_run_fixture["campaign_root"],
        expected_contract_sha256=open04_pre_run_fixture["contract_sha256"],
        compiled_dirs_by_role=open04_pre_run_fixture["compiled_dirs"],
    )

    assert contract["execution_order"] == [
        "baseline",
        "candidate_a",
        "candidate_b",
    ]
    assert contract["no_retuning"] is True
    assert contract["execution_contract"] == {
        "scenario_concurrency_mode": "parallel_distinct_output_parents",
        "maximum_concurrent_scenarios": 3,
        "aggregate_acceptance_peak_rss_bytes": 10 * 1024**3,
        "controller_placement": "auto",
        "writer_claim_scope": "output_parent",
        "parent_watchdog_required": True,
        "process_pool_enabled": False,
        "numerical_thread_count": 1,
        "minimum_available_memory_bytes": 4 * 1024**3,
        "acceptance_peak_rss_bytes": 6 * 1024**3,
        "application_abort_rss_bytes": 8 * 1024**3,
        "parent_graceful_stop_rss_bytes": 10 * 1024**3,
        "parent_kill_rss_bytes": 12 * 1024**3,
    }
    assert tuple(contract["roles"]) == (
        "baseline",
        "candidate_a",
        "candidate_b",
    )
    assert receipt["status"] == "pass"
    assert receipt["campaign_eligible"] is False
    assert receipt["promotion_status"] == "not_eligible_pre_run_only"
    assert receipt["execution_status"] == "not_started"
    assert receipt["campaign_contract_sha256"] == (
        open04_pre_run_fixture["contract_sha256"]
    )
    assert receipt["verification_order"] == [
        "baseline",
        "candidate_a",
        "candidate_b",
    ]
    assert {
        role: receipt["roles"][role]["signed_10y_shock_bp"]
        for role in receipt["roles"]
    } == {"baseline": 0.0, "candidate_a": -25.0, "candidate_b": 25.0}
    assert receipt["roles"]["candidate_a"]["economic_changed_paths"] == [
        "issuance_mix",
        "nominal_yield_curve_assumption",
        "holder_preferences",
    ]
    assert receipt["roles"]["candidate_a"]["physical_changed_inputs"] == [
        "tdcsim_holder_profile_assumptions.csv",
        "tdcsim_issuance_mix_assumptions.json",
        "tdcsim_nominal_curve_evaluated_shock.json",
    ]
    assert {
        role: receipt["roles"][role]["overall_new_issuance_wam_years"]
        for role in receipt["roles"]
    } == pytest.approx(_EXPECTED_WAM, abs=1e-12)


def test_frozen_campaign_rejects_duplicate_run_path(
    open04_pre_run_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    duplicate_paths = {
        "baseline": "same-role/same-run",
        "candidate_a": "same-role/same-run",
        "candidate_b": "candidate-b-role/candidate-b-run",
    }
    with pytest.raises(Open04CampaignError, match="must be unique"):
        freeze_open04_campaign_contract(
            tmp_path / "duplicate-campaign",
            campaign_id="duplicate-path-test",
            signature_reference="owner-ruling:test",
            scenarios_by_role=open04_pre_run_fixture["scenarios"],
            compiled_dirs_by_role=open04_pre_run_fixture["compiled_dirs"],
            code_identity=open04_pre_run_fixture["code_identity"],
            role_to_run_relative_path=duplicate_paths,
            canonical_issuance_mix_sha256_by_role=(
                open04_pre_run_fixture["mix_hashes"]
            ),
        )


def test_pre_run_receipt_rejects_contract_and_compiled_input_mutation(
    open04_pre_run_fixture: dict[str, Any],
) -> None:
    root = open04_pre_run_fixture["campaign_root"]
    contract_path = root / "open04_campaign_contract.json"
    original_contract_bytes = contract_path.read_bytes()
    original_contract = read_json(contract_path)
    tampered_contract = deepcopy(original_contract)
    tampered_contract["roles"]["candidate_a"]["signed_10y_shock_bp"] = (
        -24.0
    )
    write_json(contract_path, tampered_contract)
    with pytest.raises(Open04CampaignError):
        verify_open04_campaign_pre_run(
            root,
            expected_contract_sha256=(
                open04_pre_run_fixture["contract_sha256"]
            ),
            compiled_dirs_by_role=open04_pre_run_fixture["compiled_dirs"],
        )
    contract_path.write_bytes(original_contract_bytes)

    candidate_inputs = (
        open04_pre_run_fixture["compiled_dirs"]["candidate_b"]
        / "forecast_inputs"
    )
    target = candidate_inputs / "tdcsim_opening_runtime_state.json"
    original_bytes = target.read_bytes()
    target.write_bytes(original_bytes + b"\n")
    try:
        with pytest.raises((Open04CampaignError, VerificationError)):
            verify_open04_campaign_pre_run(
                root,
                expected_contract_sha256=(
                    open04_pre_run_fixture["contract_sha256"]
                ),
                compiled_dirs_by_role=(
                    open04_pre_run_fixture["compiled_dirs"]
                ),
            )
    finally:
        target.write_bytes(original_bytes)


def test_post_run_receipt_mapping_accepts_only_predeclared_pair(
    open04_post_receipt_fixture: dict[str, Any],
) -> None:
    verified = validate_open04_campaign_post_receipt_mapping(
        open04_post_receipt_fixture["campaign_root"],
        open04_post_receipt_fixture["receipt"],
        contract=open04_post_receipt_fixture["contract"],
        expected_contract_sha256=(
            open04_post_receipt_fixture["contract_sha256"]
        ),
    )

    assert verified["status"] == "pass"
    assert verified["campaign_eligible"] is True
    assert verified["promotion_status"] == "eligible"
    assert tuple(verified["roles"]) == (
        "baseline",
        "candidate_a",
        "candidate_b",
    )
    assert verified["pair_gates"]["terminal_sign_gate_status"] == "pass"
    assert verified["pair_gates"]["overall_wam_ordering"] == {
        "candidate_a": pytest.approx(_EXPECTED_WAM["candidate_a"]),
        "baseline": pytest.approx(_EXPECTED_WAM["baseline"]),
        "candidate_b": pytest.approx(_EXPECTED_WAM["candidate_b"]),
        "status": "pass",
    }


def test_post_run_receipt_records_common_remote_python_runtime(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    fixture = _copy_export_campaign(
        open04_export_runs_fixture,
        tmp_path / "runtime-python-campaign",
    )
    receipt = fixture["receipt"]
    runtime_version = "3.15.0"
    receipt["common_identity"]["python_version"] = runtime_version
    for role, run_root in fixture["run_roots"].items():
        manifest_path = run_root / "tdcsim_cbo_run_manifest.json"
        manifest = read_json(manifest_path)
        manifest["code_environment"]["python_version"] = runtime_version
        write_json(manifest_path, manifest)
        receipt["roles"][role]["run_manifest_sha256"] = sha256_file(
            manifest_path
        )

    verified = validate_open04_campaign_post_receipt_mapping(
        fixture["campaign_root"],
        receipt,
        contract=fixture["contract"],
        expected_contract_sha256=fixture["contract_sha256"],
    )

    assert verified["common_identity"]["python_version"] == runtime_version


def test_post_run_receipt_rejects_noncommon_remote_python_runtime(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    fixture = _copy_export_campaign(
        open04_export_runs_fixture,
        tmp_path / "mixed-runtime-python-campaign",
    )
    receipt = fixture["receipt"]
    receipt["common_identity"]["python_version"] = "3.15.0"
    for index, (role, run_root) in enumerate(
        fixture["run_roots"].items()
    ):
        manifest_path = run_root / "tdcsim_cbo_run_manifest.json"
        manifest = read_json(manifest_path)
        manifest["code_environment"]["python_version"] = (
            "3.15.0" if index < 2 else "3.16.0"
        )
        write_json(manifest_path, manifest)
        receipt["roles"][role]["run_manifest_sha256"] = sha256_file(
            manifest_path
        )

    with pytest.raises(
        Open04CampaignError,
        match="do not share one Python runtime version",
    ):
        validate_open04_campaign_post_receipt_mapping(
            fixture["campaign_root"],
            receipt,
            contract=fixture["contract"],
            expected_contract_sha256=fixture["contract_sha256"],
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_role",
        "duplicate_role",
        "baseline_mismatch",
        "producer_mismatch",
        "lock_mismatch",
        "wrong_mix",
        "wrong_shock_sign",
        "wrong_wam",
        "wrong_terminal_sign",
        "false_sign_gate",
        "blank_run_id",
        "wrong_event_schema_version",
    ),
)
def test_post_run_pair_verifier_fails_closed_on_mutations(
    open04_post_receipt_fixture: dict[str, Any],
    mutation: str,
) -> None:
    receipt = deepcopy(open04_post_receipt_fixture["receipt"])
    if mutation == "missing_role":
        del receipt["roles"]["candidate_b"]
    elif mutation == "duplicate_role":
        receipt["roles"]["candidate_b"] = deepcopy(
            receipt["roles"]["candidate_a"]
        )
    elif mutation == "baseline_mismatch":
        receipt["common_identity"]["baseline_package_sha256"] = "9" * 64
    elif mutation == "producer_mismatch":
        receipt["common_identity"]["wheel_sha256"] = "8" * 64
    elif mutation == "lock_mismatch":
        receipt["common_identity"]["requirements_lock_sha256"] = "7" * 64
    elif mutation == "wrong_mix":
        receipt["roles"]["candidate_a"][
            "canonical_issuance_mix_sha256"
        ] = "6" * 64
    elif mutation == "wrong_shock_sign":
        receipt["roles"]["candidate_a"]["signed_10y_shock_bp"] = 25.0
    elif mutation == "wrong_wam":
        receipt["roles"]["candidate_a"][
            "overall_new_issuance_wam_years"
        ] = 8.5
    elif mutation == "wrong_terminal_sign":
        receipt["roles"]["candidate_a"][
            "terminal_cumulative_tdc_change_bil"
        ] = -1.0
    elif mutation == "false_sign_gate":
        receipt["pair_gates"]["terminal_sign_gate_status"] = "fail"
    elif mutation == "blank_run_id":
        receipt["roles"]["baseline"]["run_id"] = ""
    elif mutation == "wrong_event_schema_version":
        receipt["roles"]["baseline"]["event_schema_version"] = (
            "wrong_event_schema"
        )
    else:  # pragma: no cover - guarded by parameter list
        raise AssertionError(mutation)

    with pytest.raises(Open04CampaignError):
        validate_open04_campaign_post_receipt_mapping(
            open04_post_receipt_fixture["campaign_root"],
            receipt,
            contract=open04_post_receipt_fixture["contract"],
            expected_contract_sha256=(
                open04_post_receipt_fixture["contract_sha256"]
            ),
        )


def test_post_run_pair_verifier_rejects_execution_scope_mutation(
    open04_post_receipt_fixture: dict[str, Any],
) -> None:
    contract = deepcopy(open04_post_receipt_fixture["contract"])
    contract["execution_contract"]["writer_claim_scope"] = (
        "open04_campaign_root"
    )

    with pytest.raises(Open04CampaignError):
        validate_open04_campaign_post_receipt_mapping(
            open04_post_receipt_fixture["campaign_root"],
            open04_post_receipt_fixture["receipt"],
            contract=contract,
            expected_contract_sha256=(
                open04_post_receipt_fixture["contract_sha256"]
            ),
        )


def test_post_run_campaign_recomputes_real_source_and_controller_evidence(
    open04_export_runs_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verified_run(
        run_root: str | Path,
        **_: Any,
    ) -> dict[str, Any]:
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
        verified_run,
    )
    monkeypatch.setattr(
        verifier_module,
        "_verify_code_environment",
        lambda *_args, **_kwargs: None,
    )

    baseline = open04_export_runs_fixture["baseline"]
    verified = verify_open04_campaign_post_run(
        open04_export_runs_fixture["campaign_root"],
        expected_contract_sha256=(
            open04_export_runs_fixture["contract_sha256"]
        ),
        baseline_package=baseline.package_path,
        attestation=baseline.attestation.path,
    )

    assert verified["status"] == "pass"
    assert verified["campaign_eligible"] is True
    assert verified["verification_order"] == [
        "baseline",
        "candidate_a",
        "candidate_b",
    ]
    assert verified["pair_gates"]["controller_output_isolation_status"] == "pass"
    assert verified["pair_gates"]["aggregate_memory_budget_status"] == "pass"
    assert {
        role: verified["roles"][role]["terminal_summary_sha256"]
        for role in verified["roles"]
    } == {
        role: open04_export_runs_fixture["receipt"]["roles"][role][
            "terminal_summary_sha256"
        ]
        for role in verified["roles"]
    }


@pytest.mark.parametrize("document_kind", ("mapping", "path"))
def test_thin_export_promotes_identical_exact_four_file_boundaries(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document_kind: str,
) -> None:
    _patch_export_fixture_identity(monkeypatch, open04_export_runs_fixture)
    contract_source: Any = open04_export_runs_fixture["contract"]
    receipt_source: Any = open04_export_runs_fixture["receipt"]
    if document_kind == "path":
        contract_source = (
            open04_export_runs_fixture["campaign_root"]
            / "open04_campaign_contract.json"
        )
        receipt_source = open04_export_runs_fixture["receipt_path"]

    producer = tmp_path / "producer"
    consumer = tmp_path / "consumer"
    result = export_open04_thin_package(
        open04_export_runs_fixture["run_roots"],
        contract_source,
        receipt_source,
        producer,
        consumer,
        consumer_project="ratewall",
    )

    assert result.producer_output_dir == producer.resolve()
    assert result.consumer_output_dir == consumer.resolve()
    assert tuple(sorted(path.name for path in producer.iterdir())) == tuple(
        sorted(OPEN04_EXPORT_FILES)
    )
    assert tuple(sorted(path.name for path in consumer.iterdir())) == tuple(
        sorted(OPEN04_EXPORT_FILES)
    )
    for filename in OPEN04_EXPORT_FILES:
        producer_path = producer / filename
        consumer_path = consumer / filename
        assert producer_path.read_bytes() == consumer_path.read_bytes()
        assert sha256_file(producer_path) == sha256_file(consumer_path)

    tdc_rows = _read_csv(producer / OPEN04_EXPORT_FILES[0])
    maturity_rows = _read_csv(producer / OPEN04_EXPORT_FILES[1])
    input_rows = _read_csv(producer / OPEN04_EXPORT_FILES[2])
    assert tuple(tdc_rows[0]) == TDC_PATH_COLUMNS
    assert tuple(maturity_rows[0]) == MATURITY_COLUMNS
    assert tuple(input_rows[0]) == SCENARIO_INPUT_COLUMNS
    assert len(tdc_rows) == 33
    assert len(maturity_rows) == 33
    assert len(input_rows) == 3
    assert {row["scenario_role"] for row in input_rows} == {
        "baseline",
        "candidate_a",
        "candidate_b",
    }

    first_by_role = {
        role: next(
            row
            for row in tdc_rows
            if row["scenario_role"] == role
            and row["fiscal_year"] == "2026"
        )
        for role in ("baseline", "candidate_a", "candidate_b")
    }
    for row in first_by_role.values():
        assert row["period_label"] == "FY2026_PARTIAL_OPENING"
        assert row["period_status"] == "partial_opening"
        assert row["coverage_start_exclusive"] == _START
        assert row["coverage_end_inclusive"] == "2026-09-30"
        assert row["coverage_days"] == "101"
    assert {
        int(row["fiscal_year"])
        for row in tdc_rows
        if row["scenario_role"] == "baseline"
    } == set(range(2026, 2037))

    for row in tdc_rows:
        assert float(row["tdc_change_bil"]) == pytest.approx(
            float(row["overlap_cashflow_bil"])
            + float(row["tdc_change_ex_overlap_bil"])
        )
        assert float(row["modeled_financing_cost_bil"]) == pytest.approx(
            float(row["interest_outlay_bil"])
            + float(row["issue_discount_cost_bil"])
            + float(row["nonmarketable_interest_capitalized_bil"])
            + float(row["tips_inflation_accretion_bil"])
        )
    for row in maturity_rows:
        assert row["outstanding_snapshot_date"].endswith("-09-30")
        assert row["snapshot_status"] == "exact_period_end"
        assert float(row["short_maturity_cutoff_years"]) == 1.0
        assert "cumulative" not in " ".join(row).lower()
    assert {
        row["period_status"]
        for row in tdc_rows
        if row["fiscal_year"] != "2026"
    } == {"full_fiscal_year"}

    candidate_input = next(
        row
        for row in input_rows
        if row["scenario_role"] == "candidate_a"
    )
    assert (
        candidate_input["calibration_source_label"]
        == "ATI_informed_assumed_10y_nominal_yield_level_sensitivity"
    )
    assert (
        candidate_input["calibration_status"]
        == "conditional_assumption_calibration"
    )
    assert (
        candidate_input["source_claim_boundary"]
        == "source_informs_sign_and_scenario_magnitude_not_structural_elasticity"
    )
    assert (
        candidate_input["source_range_status"]
        == "heterogeneous_method_range_not_confidence_interval"
    )
    assert candidate_input["holder_override_status"] == (
        "hand_specified_outcome_conditioned_1pp_private_to_banks_"
        "new_20y_30y_nominal_bonds"
    )
    assert "solved/chosen" in candidate_input["source_scenario_title"]

    receipt = read_json(producer / OPEN04_EXPORT_FILES[3])
    assert receipt["overall_status"] == "pass"
    assert receipt["copy_policy"] == {
        "baseline_package_copied": False,
        "bulk_run_tree_copied": False,
        "compiled_input_tree_copied": False,
        "source_outputs_copied": False,
        "source_outputs_retained_at_producer": True,
        "consumer_boundary_file_count": 4,
    }
    assert receipt["exact_file_boundary"]["expected_filenames"] == list(
        OPEN04_EXPORT_FILES
    )
    assert receipt["exact_file_boundary"]["producer_file_count"] == 4
    assert receipt["exact_file_boundary"]["consumer_file_count"] == 4
    assert receipt["exact_file_boundary"]["csv_copy_hash_status"] == "pass"
    assert (
        receipt["exact_file_boundary"]["receipt_copy_byte_identity_status"]
        == "pass"
    )
    assert receipt["receipt_authority"][
        "consumer_promoted_before_producer"
    ] is True
    assert receipt["consumer_copy_manifest_sha256"] == (
        canonical_json_sha256(receipt["consumer_copy_manifest"])
    )
    assert receipt["thin_output_manifest_sha256"] == (
        receipt["thin_artifact_manifest_sha256"]
    )
    for artifact in receipt["thin_artifacts"]:
        assert artifact["producer_sha256"] == artifact["consumer_sha256"]
        assert artifact["producer_bytes"] == artifact["consumer_bytes"]
        assert artifact["producer_rows"] == artifact["consumer_rows"]
    for scenario in receipt["scenarios"]:
        retained = {
            item["logical_name"] for item in scenario["source_outputs"]
        }
        assert {
            "controller_completion_receipt",
            "manifest_output:tdcsim_annual_economic_summary.csv.gz",
            "manifest_output:tdcsim_resource_samples.csv.gz",
            "manifest_output:summary.json",
        } <= retained
        assert scenario["memory_evidence"]["memory_watchdog_status"] == "pass"
        assert (
            scenario["accounting_evidence"][
                "accounting_invariants_status"
            ]
            == "pass"
        )
    for remote in receipt["remote_execution"]["roles"]:
        assert remote["controller_run_id"].startswith("fixture-controller-")
        assert remote["controller_command_exit_code"] == 0
        assert remote["controller_exit_code"] == 0
        assert remote["preflight_conflicts"] == 0
        assert remote["postrun_conflicts"] == 0
        assert remote["controller_process_tree_drained"] is True


def test_thin_export_accepts_host_owned_task_receipts(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _copy_export_campaign(
        open04_export_runs_fixture,
        tmp_path / "host-task-campaign",
    )
    receipt = fixture["receipt"]
    for role in ("baseline", "candidate_a", "candidate_b"):
        path = (
            fixture["campaign_root"]
            / fixture["contract"]["roles"][role][
                "controller_completion_receipt_relative_path"
            ]
        )
        controller = read_json(path)
        controller["schema_version"] = (
            OPEN04_HOST_TASK_CONTROLLER_RECEIPT_SCHEMA_VERSION
        )
        controller["controller"] = "host_owned_task"
        controller["placement"] = "auto_selected_host"
        controller["controller_avg_cpu_pct"] = None
        controller["controller_telemetry_status"] = (
            "bounded_role_watchdogs"
        )
        controller["controller_telemetry_locator"] = (
            "controller_summary_sha256:"
            f"{controller['controller_summary_sha256']}"
        )
        write_json(path, controller)
        receipt["roles"][role][
            "controller_completion_receipt_sha256"
        ] = sha256_file(path)
    _patch_export_fixture_identity(monkeypatch, fixture)

    result = export_open04_thin_package(
        fixture["run_roots"],
        fixture["contract"],
        receipt,
        tmp_path / "producer-host-task",
        tmp_path / "consumer-host-task",
        consumer_project="ratewall",
        exporter_code_commit_sha="6" * 40,
    )

    assert result.receipt["overall_status"] == "pass"
    assert result.receipt["exporter"]["identity_mode"] == (
        "committed_post_run_adapter"
    )
    assert result.receipt["exporter"]["code_commit_sha"] == "6" * 40
    assert {
        item["controller"]
        for item in result.receipt["remote_execution"]["roles"]
    } == {"host_owned_task"}


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("clock", "annual clock contract"),
        ("tdc_identity", "TDC identity"),
        ("financing_identity", "financing-cost identity"),
        ("maturity_identity", "new_issuance_wam_years"),
    ),
)
def test_thin_export_rejects_annual_contract_mutations_after_rehash(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    fixture = _copy_export_campaign(
        open04_export_runs_fixture,
        tmp_path / "mutated-campaign",
    )
    receipt = deepcopy(fixture["receipt"])
    annual_path = (
        fixture["run_roots"]["candidate_a"]
        / "outputs"
        / "tdcsim_annual_economic_summary.csv.gz"
    )
    rows = _read_gzip_csv(annual_path)
    if mutation == "clock":
        rows[0]["coverage_days"] = "100"
    elif mutation == "tdc_identity":
        rows[-1]["tdc_change_ex_overlap_bil"] = "7.0"
        rows[-1]["cumulative_tdc_change_ex_overlap_bil"] = "7.0"
    elif mutation == "financing_identity":
        rows[-1]["modeled_financing_cost_bil"] = "98.0"
        rows[-1]["cumulative_modeled_financing_cost_bil"] = "98.0"
    elif mutation == "maturity_identity":
        rows[-1]["new_issuance_wam_years"] = "9.0"
    else:  # pragma: no cover - guarded by parameter list
        raise AssertionError(mutation)
    _write_gzip_csv(annual_path, ANNUAL_COLUMNS, rows)
    _refresh_export_run_manifest(
        fixture["run_roots"]["candidate_a"],
        receipt["roles"]["candidate_a"],
    )
    _patch_export_fixture_identity(monkeypatch, fixture)

    with pytest.raises(Open04ExportError, match=match):
        export_open04_thin_package(
            fixture["run_roots"],
            fixture["contract"],
            receipt,
            tmp_path / "producer",
            tmp_path / "consumer",
            consumer_project="ratewall",
        )


def test_thin_export_rejects_destination_conflicts_without_partial_output(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_export_fixture_identity(monkeypatch, open04_export_runs_fixture)
    same = tmp_path / "same"
    with pytest.raises(Open04ExportError, match="must be distinct"):
        export_open04_thin_package(
            open04_export_runs_fixture["run_roots"],
            open04_export_runs_fixture["contract"],
            open04_export_runs_fixture["receipt"],
            same,
            same,
            consumer_project="ratewall",
        )
    assert not same.exists()

    producer = tmp_path / "preexisting-producer"
    producer.mkdir()
    consumer = tmp_path / "new-consumer"
    with pytest.raises(Open04ExportError, match="must not already exist"):
        export_open04_thin_package(
            open04_export_runs_fixture["run_roots"],
            open04_export_runs_fixture["contract"],
            open04_export_runs_fixture["receipt"],
            producer,
            consumer,
            consumer_project="ratewall",
        )
    assert not consumer.exists()
    assert list(producer.iterdir()) == []

    inside_campaign = (
        open04_export_runs_fixture["campaign_root"] / "thin-export"
    )
    with pytest.raises(Open04ExportError, match="overlaps"):
        export_open04_thin_package(
            open04_export_runs_fixture["run_roots"],
            open04_export_runs_fixture["contract"],
            open04_export_runs_fixture["receipt"],
            inside_campaign,
            tmp_path / "outside-consumer",
            consumer_project="ratewall",
        )
    assert not inside_campaign.exists()
    assert not (tmp_path / "outside-consumer").exists()


def test_thin_export_rolls_back_consumer_when_producer_promotion_fails(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_export_fixture_identity(monkeypatch, open04_export_runs_fixture)
    producer = (tmp_path / "producer").resolve()
    consumer = (tmp_path / "consumer").resolve()
    original_rename = Path.rename

    def fail_producer_rename(source: Path, target: str | Path) -> Path:
        if Path(target).resolve() == producer:
            raise OSError("synthetic producer promotion failure")
        return original_rename(source, target)

    monkeypatch.setattr(Path, "rename", fail_producer_rename)
    with pytest.raises(OSError, match="synthetic producer promotion"):
        export_open04_thin_package(
            open04_export_runs_fixture["run_roots"],
            open04_export_runs_fixture["contract"],
            open04_export_runs_fixture["receipt"],
            producer,
            consumer,
            consumer_project="ratewall",
        )

    assert not producer.exists()
    assert not consumer.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_thin_export_cleans_first_staging_when_second_creation_fails(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_export_fixture_identity(monkeypatch, open04_export_runs_fixture)
    producer = (tmp_path / "producer").resolve()
    consumer = (tmp_path / "consumer").resolve()
    original_mkdir = Path.mkdir

    def fail_consumer_staging(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        if path.parent == consumer.parent and path.name.startswith(
            ".consumer.staging-"
        ):
            raise OSError("synthetic consumer staging creation failure")
        original_mkdir(
            path,
            mode=mode,
            parents=parents,
            exist_ok=exist_ok,
        )

    monkeypatch.setattr(Path, "mkdir", fail_consumer_staging)
    with pytest.raises(
        OSError,
        match="synthetic consumer staging creation",
    ):
        export_open04_thin_package(
            open04_export_runs_fixture["run_roots"],
            open04_export_runs_fixture["contract"],
            open04_export_runs_fixture["receipt"],
            producer,
            consumer,
            consumer_project="ratewall",
        )
    assert not producer.exists()
    assert not consumer.exists()
    assert not list(tmp_path.glob(".*.staging-*"))


def test_marked_baseline_runner_requires_parent_watchdog(
    tmp_path: Path,
) -> None:
    spec = CboScenarioSpec.from_mapping(_marked_baseline())

    with pytest.raises(RunnerError, match="parent RSS watchdog"):
        run_cbo_scenario(
            SimpleNamespace(),
            spec,
            tmp_path / "baseline-run",
        )


def test_marked_baseline_runner_requires_release_identity_before_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CboScenarioSpec.from_mapping(_marked_baseline())
    _pin_threads(monkeypatch)
    out = tmp_path / "baseline-run"
    handoff = tmp_path / ".baseline-run.watchdog-handoff-test.json"

    with pytest.raises(RunnerError, match="retained release wheel"):
        run_cbo_scenario(
            SimpleNamespace(
                attestation=SimpleNamespace(data={}),
            ),
            spec,
            out,
            watchdog_handoff=handoff,
        )


def test_marked_baseline_runner_requires_campaign_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = CboScenarioSpec.from_mapping(_marked_baseline())
    _pin_threads(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_assert_open04_release_identity",
        lambda _baseline: {},
    )
    monkeypatch.delenv("TDCSIM_CBO_CAMPAIGN_ROOT", raising=False)
    monkeypatch.delenv("TDCSIM_CBO_CAMPAIGN_ID", raising=False)
    out = tmp_path / "baseline-run"
    handoff = tmp_path / ".baseline-run.watchdog-handoff-test.json"

    with pytest.raises(RunnerError, match="campaign root and ID"):
        run_cbo_scenario(
            SimpleNamespace(),
            spec,
            out,
            watchdog_handoff=handoff,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("marker", "campaign marker"),
        ("simulation", "simulation dates"),
        ("output", "output manifest"),
    ),
)
def test_marked_baseline_verifier_rejects_manifest_scope_tamper(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    run_root = tmp_path / "run"
    compiled_dir = run_root / "compile"
    compiled_dir.mkdir(parents=True)
    scenario_path = run_root / "scenario.json"
    write_json(scenario_path, _marked_baseline())
    write_json(
        compiled_dir / "tdcsim_cbo_compiled_manifest.json",
        {
            "open04_simulation_contract": {
                "start_date": _START,
                "end_date": _END,
                "frequency": "daily",
                "output_profile": "compact",
                "compression": "gzip",
            }
        },
    )
    manifest = {
        "scenario": {"relative_path": scenario_path.name},
        "open04_campaign": {
            "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
            "role": "baseline",
            "funding_closure_mode": OPEN04_FUNDING_CLOSURE_MODE,
        },
        "simulation": {
            "start_date": _START,
            "end_date": _END,
            "frequency": "daily",
        },
        "output_manifest": {"profile": "compact", "compression": "gzip"},
    }
    if mutation == "marker":
        manifest["open04_campaign"]["role"] = "candidate_a"
    elif mutation == "simulation":
        manifest["simulation"]["end_date"] = "2036-09-29"
    elif mutation == "output":
        manifest["output_manifest"]["compression"] = "none"
    else:  # pragma: no cover - guarded by parameter list
        raise AssertionError(mutation)

    with pytest.raises(VerificationError, match=match):
        _verify_run_evaluated_nominal_curve(
            compiled_dir,
            run_root,
            manifest,
            compiled_curve=None,
        )


def test_campaign_contract_allows_a_declared_post_run_exporter_adapter(
    open04_export_runs_fixture: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = open04_export_runs_fixture
    original_sha256_file = open04_export_module.sha256_file

    def adapter_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved == Path(open04_export_module.__file__).resolve():
            return "9" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(
        open04_export_module, "sha256_file", adapter_sha256
    )

    validated = open04_export_module._validate_campaign_contract(
        fixture["contract"],
        roots=fixture["run_roots"],
        campaign_root=fixture["campaign_root"],
        frozen_sha256=fixture["contract_sha256"],
    )

    assert validated["campaign_id"] == fixture["contract"]["campaign_id"]


def test_thin_export_accepts_complete_cached_bounded_replay_results(
    open04_export_runs_fixture: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _copy_export_campaign(
        open04_export_runs_fixture,
        tmp_path / "cached-verification-campaign",
    )
    _patch_export_fixture_identity(monkeypatch, fixture)
    monkeypatch.setattr(
        open04_export_module,
        "verify_scenario_run",
        lambda _root: pytest.fail("unexpected duplicate run verification"),
    )
    cached = {
        role: {
            "status": "pass",
            "verification_grade": "bounded_replay_v1",
        }
        for role in ("baseline", "candidate_a", "candidate_b")
    }

    result = export_open04_thin_package(
        fixture["run_roots"],
        fixture["contract"],
        fixture["receipt"],
        tmp_path / "producer-cached",
        tmp_path / "consumer-cached",
        consumer_project="ratewall",
        exporter_code_commit_sha="6" * 40,
        verified_run_results=cached,
    )

    assert result.receipt["overall_status"] == "pass"


def test_marked_baseline_verifier_accepts_explicit_funding_closure_mode(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    compiled_dir = run_root / "compile"
    compiled_dir.mkdir(parents=True)
    scenario = _marked_baseline()
    scenario["open04_campaign"]["funding_closure_mode"] = (
        OPEN04_FUNDING_CLOSURE_MODE
    )
    scenario_path = run_root / "scenario.json"
    write_json(scenario_path, scenario)
    write_json(
        compiled_dir / "tdcsim_cbo_compiled_manifest.json",
        {
            "open04_simulation_contract": {
                "start_date": _START,
                "end_date": _END,
                "frequency": "daily",
                "output_profile": "compact",
                "compression": "gzip",
            }
        },
    )
    _verify_run_evaluated_nominal_curve(
        compiled_dir,
        run_root,
        {
            "scenario": {"relative_path": scenario_path.name},
            "open04_campaign": {
                "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
                "role": "baseline",
                "funding_closure_mode": OPEN04_FUNDING_CLOSURE_MODE,
            },
            "simulation": {
                "start_date": _START,
                "end_date": _END,
                "frequency": "daily",
            },
            "output_manifest": {
                "profile": "compact",
                "compression": "gzip",
            },
        },
        compiled_curve=None,
    )


def test_marked_baseline_verifier_requires_watchdog_and_campaign_scope(
    tmp_path: Path,
) -> None:
    root, manifest, _results, _rows = _package(tmp_path)
    manifest["open04_campaign"] = {
        "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
        "role": "baseline",
    }
    with pytest.raises(VerificationError, match="parent watchdog acceptance"):
        _verify_bounded_manifest_contract(root, manifest)

    _add_parent_watchdog_acceptance(manifest)
    limits = BoundedResourceLimits()
    manifest["bounded_evidence"]["memory_thresholds"] = {
        "minimum_available_bytes": limits.minimum_available_bytes,
        "acceptance_peak_rss_bytes": limits.acceptance_peak_rss_bytes,
        "application_abort_rss_bytes": limits.application_abort_rss_bytes,
        "parent_graceful_stop_rss_bytes": (
            limits.parent_graceful_stop_rss_bytes
        ),
        "parent_kill_rss_bytes": limits.parent_kill_rss_bytes,
    }
    manifest["execution_contract"]["writer_claim_scope"] = "output_parent"
    manifest["execution_contract"]["writer_claim_scope_id"] = (
        "wrong-output-parent"
    )
    with pytest.raises(VerificationError, match="writer_claim_scope"):
        _verify_bounded_manifest_contract(root, manifest)


def _candidate_mixes() -> dict[str, dict[str, Any]]:
    return {
        "candidate_a": {
            "mode": "replace_shares",
            "tips_share": 0.06,
            "frn_share": 0.04,
            "fixed_remainder_shares": {
                "bills": 0.335030,
                "notes": 0.507485,
                "bonds": 0.157485,
            },
            "maturity_distributions": {
                "bills": [
                    {"maturity_years": 0.25, "share": 0.141717},
                    {"maturity_years": 0.5, "share": 0.858283},
                ],
                "notes": [
                    {"maturity_years": 2.0, "share": 0.198404},
                    {"maturity_years": 5.0, "share": 0.801596},
                ],
                "bonds": [
                    {"maturity_years": 20.0, "share": 0.858283},
                    {"maturity_years": 30.0, "share": 0.141717},
                ],
                "tips": [{"maturity_years": 10.0, "share": 1.0}],
                "frn": [{"maturity_years": 2.0, "share": 1.0}],
            },
            "negative_issuance_action": "error",
        },
        "candidate_b": {
            "mode": "replace_shares",
            "tips_share": 0.06,
            "frn_share": 0.04,
            "fixed_remainder_shares": {
                "bills": 0.228928,
                "notes": 0.535952,
                "bonds": 0.235120,
            },
            "maturity_distributions": {
                "bills": [{"maturity_years": 0.5, "share": 1.0}],
                "notes": [
                    {"maturity_years": 5.0, "share": 0.859522},
                    {"maturity_years": 7.0, "share": 0.063215},
                    {"maturity_years": 10.0, "share": 0.077263},
                ],
                "bonds": [
                    {"maturity_years": 20.0, "share": 0.908689},
                    {"maturity_years": 30.0, "share": 0.091311},
                ],
                "tips": [{"maturity_years": 10.0, "share": 1.0}],
                "frn": [{"maturity_years": 2.0, "share": 1.0}],
            },
            "negative_issuance_action": "error",
        },
    }


@pytest.fixture(scope="module")
def open04_pre_run_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    tmp_path = tmp_path_factory.mktemp("open04-pre-run")
    baseline = _full_horizon_compiler_baseline(tmp_path)
    baseline_identity = {
        "package_id": baseline.package_id,
        "package_sha256": baseline.package_sha256,
        "manifest_sha256": baseline.manifest_sha256,
        "release_attestation_sha256": baseline.attestation.sha256,
    }
    original_baseline_identity = open04_campaign_module.OPEN04_BASELINE_IDENTITY
    open04_campaign_module.OPEN04_BASELINE_IDENTITY = baseline_identity
    mixes = _candidate_mixes()
    scenarios = build_open04_scenario_mappings(
        baseline_identity=baseline_identity,
        candidate_a_issuance_mix=mixes["candidate_a"],
        candidate_b_issuance_mix=mixes["candidate_b"],
    )
    compiled_dirs: dict[str, Path] = {}
    for role, scenario in scenarios.items():
        compiled = CboScenarioCompiler().compile(
            baseline,
            CboScenarioSpec.from_mapping(scenario),
            tmp_path / f"compiled-{role}",
        )
        compiled_dirs[role] = compiled.compiled_dir
    mix_hashes = {
        role: canonical_json_sha256(
            read_json(
                compiled_dir
                / "forecast_inputs"
                / "tdcsim_issuance_mix_assumptions.json"
            )
        )
        for role, compiled_dir in compiled_dirs.items()
    }
    code_identity = _code_identity()
    campaign_root = tmp_path / "campaign"
    contract = freeze_open04_campaign_contract(
        campaign_root,
        campaign_id="open04-fixture-campaign",
        signature_reference="owner-ruling:test",
        scenarios_by_role=scenarios,
        compiled_dirs_by_role=compiled_dirs,
        code_identity=code_identity,
        role_to_run_relative_path={
            "baseline": "baseline-role/baseline-run",
            "candidate_a": "candidate-a-role/candidate-a-run",
            "candidate_b": "candidate-b-role/candidate-b-run",
        },
        canonical_issuance_mix_sha256_by_role=mix_hashes,
    )
    contract_path = campaign_root / "open04_campaign_contract.json"
    try:
        yield {
            "baseline": baseline,
            "campaign_root": campaign_root,
            "contract": contract,
            "contract_sha256": sha256_file(contract_path),
            "compiled_dirs": compiled_dirs,
            "scenarios": scenarios,
            "mix_hashes": mix_hashes,
            "code_identity": code_identity,
        }
    finally:
        open04_campaign_module.OPEN04_BASELINE_IDENTITY = (
            original_baseline_identity
        )


@pytest.fixture(scope="module")
def open04_post_receipt_fixture(
    open04_pre_run_fixture: dict[str, Any],
) -> dict[str, Any]:
    receipt = _synthetic_post_receipt(
        open04_pre_run_fixture["contract"],
        open04_pre_run_fixture["contract_sha256"],
    )
    return {
        **open04_pre_run_fixture,
        "receipt": receipt,
    }


@pytest.fixture(scope="module")
def open04_export_runs_fixture(
    open04_post_receipt_fixture: dict[str, Any],
) -> dict[str, Any]:
    fixture = open04_post_receipt_fixture
    contract = fixture["contract"]
    receipt = fixture["receipt"]
    campaign_root = fixture["campaign_root"]
    run_roots: dict[str, Path] = {}
    for role in ("baseline", "candidate_a", "candidate_b"):
        root = campaign_root / contract["roles"][role]["run_relative_path"]
        root.mkdir(parents=True)
        run_roots[role] = root
        _materialize_export_run(
            role,
            root,
            fixture["compiled_dirs"][role],
            campaign_root
            / contract["roles"][role]["scenario_source_relative_path"],
            contract=contract,
            receipt_role=receipt["roles"][role],
        )

    for role in ("baseline", "candidate_a", "candidate_b"):
        controller_index = list(run_roots).index(role)
        controller_run_id = "fixture-controller-batch"
        controller_path = (
            campaign_root
            / contract["roles"][role][
                "controller_completion_receipt_relative_path"
            ]
        )
        write_json(
            controller_path,
            {
                "schema_version": OPEN04_CONTROLLER_RECEIPT_SCHEMA_VERSION,
                "campaign_id": contract["campaign_id"],
                "campaign_contract_sha256": (
                    fixture["contract_sha256"]
                ),
                "role": role,
                "run_id": receipt["roles"][role]["run_id"],
                "host": "BZOT",
                "controller": "remote_controller",
                "placement": "auto",
                "terminal_status": "completed",
                "started_at_utc": "2026-07-30T00:00:00Z",
                "completed_at_utc": "2026-07-30T03:00:00Z",
                "worker_exit_confirmed_at_utc": "2026-07-30T03:01:00Z",
                "terminal_summary_sha256": receipt["roles"][role][
                    "terminal_summary_sha256"
                ],
                "run_manifest_sha256": receipt["roles"][role][
                    "run_manifest_sha256"
                ],
                "controller_run_id": controller_run_id,
                "controller_summary_sha256": "4" * 64,
                "controller_command_sha256": "7" * 64,
                "controller_command_exit_code": 0,
                "controller_exit_code": 0,
                "preflight_conflicts": 0,
                "postrun_conflicts": 0,
                "controller_peak_rss_mb": 1536.0,
                "controller_avg_cpu_pct": 240.0,
                "controller_memory_guard_status": "pass",
                "controller_telemetry_status": "reported",
                "controller_telemetry_locator": (
                    f"controller_run_id:{controller_run_id}"
                ),
                "controller_process_tree_drained": True,
            },
        )
        receipt["roles"][role][
            "controller_completion_receipt_sha256"
        ] = sha256_file(controller_path)
    receipt_path = campaign_root / "open04_campaign_verification_receipt.json"
    write_json(receipt_path, receipt)
    return {
        **fixture,
        "run_roots": run_roots,
        "receipt_path": receipt_path,
    }


def _materialize_export_run(
    role: str,
    root: Path,
    compiled_dir: Path,
    scenario_source: Path,
    *,
    contract: dict[str, Any],
    receipt_role: dict[str, Any],
) -> None:
    scenario_path = root / "scenario.json"
    shutil.copyfile(scenario_source, scenario_path)
    receipt_role["run_scenario_copy_sha256"] = sha256_file(scenario_path)

    compiled_root = root / "compile" / "compiled"
    shutil.copytree(compiled_dir, compiled_root)
    shutil.copytree(
        compiled_dir.parent / "baseline",
        root / "compile" / "baseline",
    )
    compiled_target = compiled_root / "forecast_inputs"
    compiled_manifest = read_json(
        compiled_root / "tdcsim_cbo_compiled_manifest.json"
    )
    logical_names = [
        record["path"] for record in compiled_manifest["input_hashes"]
    ]
    compiled_records: list[dict[str, Any]] = []
    for logical_name in logical_names:
        target = compiled_target / logical_name
        compiled_records.append(
            {
                "logical_name": logical_name,
                "relative_path": (
                    f"compile/compiled/forecast_inputs/{logical_name}"
                ),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            }
        )

    issuance_mix = read_json(
        compiled_target / "tdcsim_issuance_mix_assumptions.json"
    )
    annual_rows = _annual_rows(
        role,
        issuance_mix,
        terminal_cost=float(
            receipt_role[
                "terminal_cumulative_modeled_financing_cost_bil"
            ]
        ),
        terminal_tdc=float(
            receipt_role["terminal_cumulative_tdc_change_bil"]
        ),
    )
    outputs = root / "outputs"
    outputs.mkdir()
    annual_path = outputs / "tdcsim_annual_economic_summary.csv.gz"
    _write_gzip_csv(annual_path, ANNUAL_COLUMNS, annual_rows)
    summary_path = outputs / "summary.json"
    write_json(
        summary_path,
        {
            "status": "complete",
            "run_id": receipt_role["run_id"],
            "scenario_role": role,
        },
    )
    receipt_role["terminal_summary_sha256"] = sha256_file(summary_path)
    resource_path = outputs / "tdcsim_resource_samples.csv.gz"
    _write_gzip_csv(
        resource_path,
        RESOURCE_COLUMNS,
        [
            {
                column: {
                    "sample_kind": "final",
                    "period_end": _END,
                    "rss_bytes": 256 * 1024**2,
                    "peak_rss_bytes": 512 * 1024**2,
                    "host_available_bytes": 24 * 1024**3,
                    "process_cpu_seconds": 1.0,
                    "portfolio_rows": 10,
                    "active_portfolio_rows": 10,
                    "event_count": receipt_role["event_count"],
                    "current_key_cardinality": 10,
                    "max_key_cardinality": 10,
                    "bytes_written": annual_path.stat().st_size,
                }.get(column, "")
                for column in RESOURCE_COLUMNS
            }
        ],
    )

    def output_record(
        path: Path,
        *,
        row_count: int | None = None,
    ) -> dict[str, Any]:
        record = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if row_count is not None:
            record["row_count"] = row_count
        return record

    annual_record = output_record(
        annual_path,
        row_count=len(annual_rows),
    )
    resource_record = output_record(resource_path, row_count=1)
    summary_record = output_record(summary_path)

    common = contract["common_identity"]
    output_manifest = {
        "profile": "compact",
        "compression": "gzip",
        "evidence_profile": "bounded_period_closure_v1",
        "verification_grade": "bounded_replay_v1",
        "annual": deepcopy(annual_record),
        "resources": deepcopy(resource_record),
        "summary": deepcopy(summary_record),
        "deterministic_evidence_artifacts": {
            "annual": deepcopy(annual_record)
        },
        "row_metadata": {
            "scenario_id": receipt_role["scenario_id"],
            "run_id": receipt_role["run_id"],
            "package_id": common["package_id"],
            "actuals_available_as_of": "2026-06-20",
            "scenario_config_sha256": receipt_role["scenario_sha256"],
            "compiled_inputs_digest": receipt_role[
                "compiled_inputs_digest"
            ],
            "mmf_deposit_pass_through": 1.0,
            "mmf_deposit_pass_through_status": "fixed_baseline_input",
            "fiscal_incidence_policy_id": "tdcsim_fiscal_incidence_v1",
            "fiscal_incidence_basis": "cash_settlement",
            "fiscal_incidence_du_share": 1.0,
        },
    }
    receipt_role["source_output_manifest_sha256"] = canonical_json_sha256(
        output_manifest
    )
    code_environment = {
        field: common[field]
        for field in open04_campaign_module._CODE_IDENTITY_KEYS
        if field != "wheel_artifact_sha256"
    }
    code_environment["wheel_artifact"] = {
        "sha256": common["wheel_artifact_sha256"]
    }
    code_environment["producer_source_identity"] = {
        "installed_archive_sha256": common["wheel_sha256"],
        "source_tree": {
            "release_commit_sha": common["code_commit_sha"],
            "dependency_lock_files": [
                {
                    "relative_path": "uv.lock",
                    "sha256": common["uv_lock_sha256"],
                    "bytes": 10,
                },
                {
                    "relative_path": "requirements.lock.txt",
                    "sha256": common["requirements_lock_sha256"],
                    "bytes": 20,
                },
            ],
            "dependency_lock_set_sha256": common[
                "dependency_lock_set_sha256"
            ],
        },
    }
    manifest = {
        "schema_version": "tdcsim_cbo_scenario_run_manifest_v2",
        "status": "complete",
        "verification_grade": "bounded_replay_v1",
        "evidence_profile": "bounded_period_closure_v1",
        "run_id": receipt_role["run_id"],
        "baseline": {
            "package_id": common["package_id"],
            "package_sha256": common["baseline_package_sha256"],
            "manifest_sha256": common["baseline_manifest_sha256"],
            "release_attestation_sha256": common[
                "release_attestation_sha256"
            ],
        },
        "scenario": {
            "scenario_id": receipt_role["scenario_id"],
            "canonical_sha256": receipt_role["scenario_sha256"],
            "source_file_sha256": sha256_file(scenario_path),
            "relative_path": scenario_path.name,
        },
        "open04_campaign": {
            "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
            "role": role,
            "funding_closure_mode": (
                "cbo_debt_reference_plus_tga_floor_financing_v1"
            ),
        },
        "coupling_decisions": dict(OPEN04_FIXED_COUPLING),
        "compiled_manifest": (
            "compile/compiled/tdcsim_cbo_compiled_manifest.json"
        ),
        "compiled_inputs_digest": receipt_role["compiled_inputs_digest"],
        "compiled_inputs": compiled_records,
        "simulation": {
            "start_date": _START,
            "end_date": _END,
            "frequency": "daily",
        },
        "aggregation_clock": {
            "clock_id": "federal_fiscal_year_period_end_v1"
        },
        "bounded_evidence": {
            "invariant_status": "pass",
            "event_schema_version": receipt_role[
                "event_schema_version"
            ],
            "event_count": receipt_role["event_count"],
            "event_root_sha256": receipt_role["event_root_sha256"],
            "peak_rss_bytes": 512 * 1024**2,
            "memory_thresholds": {
                "minimum_available_bytes": 4 * 1024**3,
                "acceptance_peak_rss_bytes": 6 * 1024**3,
                "application_abort_rss_bytes": 8 * 1024**3,
                "parent_graceful_stop_rss_bytes": 10 * 1024**3,
                "parent_kill_rss_bytes": 12 * 1024**3,
            },
            "deterministic_artifacts": {
                "annual": deepcopy(annual_record)
            },
            "resource_artifact": deepcopy(resource_record),
        },
        "output_manifest": output_manifest,
        "outputs": [
            {
                "logical_name": path.name,
                "relative_path": f"outputs/{path.name}",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "media_type": (
                    "application/json"
                    if path.suffix == ".json"
                    else "text/csv"
                ),
            }
            for path in (annual_path, resource_path, summary_path)
        ],
        "code_environment": code_environment,
        "execution_contract": {
            "schema_version": "tdcsim_cbo_execution_contract_v1",
            "single_writer_claim": True,
            "writer_claim_file_name": ".tdcsim-cbo-bounded-writer.claim",
            "writer_claim_scope": "output_parent",
            "writer_claim_scope_id": (
                f"{OPEN04_CAMPAIGN_CONTRACT_ID}.{role}"
            ),
            "one_scenario_per_worker": True,
            "process_pool_enabled": False,
            "parent_watchdog_required": True,
            "scenario_process_mode": "parent_watchdog_worker",
            "numerical_thread_environment": {
                name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
            },
        },
        "parent_watchdog": {
            "status": "accepted",
            "sampler": "parent_process_rss_poll_v1",
            "child_pid": 1234,
            "child_returncode": 0,
            "action": "completed",
            "peak_rss_bytes": 512 * 1024**2,
            "worker_peak_rss_bytes": 512 * 1024**2,
            "effective_peak_rss_bytes": 512 * 1024**2,
            "acceptance_peak_rss_bytes": 6 * 1024**3,
            "terminate_rss_bytes": 10 * 1024**3,
            "kill_rss_bytes": 12 * 1024**3,
            "poll_interval_seconds": 1.0,
        },
    }
    if role != "baseline":
        manifest["evaluated_nominal_curve"] = {
            "sidecar_sha256": receipt_role["curve_sidecar_sha256"],
            "evaluated_delta_sha256": receipt_role[
                "compiled_curve_delta_digest"
            ],
            "short_end_bitwise_mismatch_count": 0,
            "max_abs_analytic_delta_error_decimal": 0.0,
        }
    manifest_path = root / "tdcsim_cbo_run_manifest.json"
    write_json(manifest_path, manifest)
    receipt_role["run_manifest_sha256"] = sha256_file(manifest_path)


def _annual_rows(
    role: str,
    issuance_mix: dict[str, Any],
    *,
    terminal_cost: float,
    terminal_tdc: float,
) -> list[dict[str, Any]]:
    shares = issuance_mix["security_shares"]
    distributions = issuance_mix["maturity_distributions"]
    issuance_face = 10.0
    wam = float(issuance_mix["weighted_average_maturity_years"])
    bill_share = float(shares["bills"])
    short_share = 0.0
    for security_type, security_share in shares.items():
        within_short = sum(
            float(item["share"])
            for item in distributions[security_type]
            if float(item["maturity_years"]) <= 1.0
        )
        short_share += float(security_share) * within_short

    rows: list[dict[str, Any]] = []
    cumulative = {
        "tdc": 0.0,
        "overlap": 0.0,
        "ex_overlap": 0.0,
        "interest": 0.0,
        "discount": 0.0,
        "nonmarketable": 0.0,
        "tips": 0.0,
        "cost": 0.0,
    }
    for year in range(2026, 2037):
        final = year == 2036
        tdc = terminal_tdc if final else 0.0
        overlap = tdc * 0.4
        ex_overlap = tdc - overlap
        cost = terminal_cost if final else 0.0
        interest = cost * 0.4
        discount = cost * 0.3
        nonmarketable = cost * 0.2
        tips = cost - interest - discount - nonmarketable
        for key, value in (
            ("tdc", tdc),
            ("overlap", overlap),
            ("ex_overlap", ex_overlap),
            ("interest", interest),
            ("discount", discount),
            ("nonmarketable", nonmarketable),
            ("tips", tips),
            ("cost", cost),
        ):
            cumulative[key] += value
        period_start = (
            "2026-06-21" if year == 2026 else f"{year - 1}-09-30"
        )
        period_end = f"{year}-09-30"
        start_date = date.fromisoformat(period_start)
        end_date = date.fromisoformat(period_end)
        full_start = date(year - 1, 9, 30)
        rows.append(
            {
                "period_label": (
                    "FY2026_PARTIAL_OPENING"
                    if year == 2026
                    else f"FY{year}"
                ),
                "period_start": period_start,
                "period_end": period_end,
                "is_partial_period": "true" if year == 2026 else "false",
                "coverage_days": (end_date - start_date).days,
                "expected_coverage_days": (end_date - full_start).days,
                "aggregation_clock_id": (
                    "federal_fiscal_year_period_end_v1"
                ),
                "tdc_change_bil": tdc,
                "overlap_cashflow_bil": overlap,
                "tdc_change_ex_overlap_bil": ex_overlap,
                "cumulative_tdc_change_bil": cumulative["tdc"],
                "cumulative_overlap_cashflow_bil": cumulative["overlap"],
                "cumulative_tdc_change_ex_overlap_bil": cumulative[
                    "ex_overlap"
                ],
                "interest_outlay_bil": interest,
                "issue_discount_cost_bil": discount,
                "nonmarketable_interest_capitalized_bil": nonmarketable,
                "tips_inflation_accretion_bil": tips,
                "modeled_financing_cost_bil": cost,
                "cumulative_interest_outlay_bil": cumulative["interest"],
                "cumulative_issue_discount_cost_bil": cumulative[
                    "discount"
                ],
                "cumulative_nonmarketable_interest_capitalized_bil": (
                    cumulative["nonmarketable"]
                ),
                "cumulative_tips_inflation_accretion_bil": cumulative[
                    "tips"
                ],
                "cumulative_modeled_financing_cost_bil": cumulative[
                    "cost"
                ],
                "modeled_financing_cost_basis": (
                    "nominal_model_cost_incurred_within_simulation_horizon"
                ),
                "modeled_financing_cost_units": (
                    "billions_of_nominal_dollars"
                ),
                "cumulative_basis": "since_simulation_origin",
                "new_issuance_face_bil": issuance_face,
                "new_issuance_original_term_face_years_bil": (
                    issuance_face * wam
                ),
                "new_issuance_bill_face_bil": issuance_face * bill_share,
                "new_issuance_short_face_bil": issuance_face * short_share,
                "new_issuance_wam_years": wam,
                "new_issuance_bill_share": bill_share,
                "new_issuance_short_maturity_share": short_share,
                "snapshot_date": period_end,
                "outstanding_controlled_wam_years": (
                    6.0 if role == "baseline" else (5.5 if role == "candidate_a" else 6.5)
                ),
                "outstanding_controlled_bill_share": (
                    0.25 if role == "baseline" else (0.3 if role == "candidate_a" else 0.2)
                ),
                "outstanding_controlled_short_maturity_share": (
                    0.2 if role == "baseline" else (0.25 if role == "candidate_a" else 0.15)
                ),
            }
        )
    return rows


def _refresh_export_run_manifest(
    run_root: Path,
    receipt_role: dict[str, Any],
) -> None:
    manifest_path = run_root / "tdcsim_cbo_run_manifest.json"
    manifest = read_json(manifest_path)
    annual_path = (
        run_root
        / "outputs"
        / "tdcsim_annual_economic_summary.csv.gz"
    )
    manifest["bounded_evidence"]["deterministic_artifacts"]["annual"] = {
        "path": annual_path.name,
        "sha256": sha256_file(annual_path),
        "bytes": annual_path.stat().st_size,
        "row_count": 11,
    }
    manifest["output_manifest"]["annual"] = deepcopy(
        manifest["bounded_evidence"]["deterministic_artifacts"]["annual"]
    )
    manifest["output_manifest"]["deterministic_evidence_artifacts"][
        "annual"
    ] = deepcopy(
        manifest["bounded_evidence"]["deterministic_artifacts"]["annual"]
    )
    for record in manifest["outputs"]:
        if record["relative_path"] == (
            "outputs/tdcsim_annual_economic_summary.csv.gz"
        ):
            record["sha256"] = sha256_file(annual_path)
            record["bytes"] = annual_path.stat().st_size
            break
    receipt_role["source_output_manifest_sha256"] = canonical_json_sha256(
        manifest["output_manifest"]
    )
    write_json(manifest_path, manifest)
    receipt_role["run_manifest_sha256"] = sha256_file(manifest_path)
    matching_controllers = []
    for controller_path in run_root.parent.parent.glob(
        "open04_*_controller_completion.json"
    ):
        controller = read_json(controller_path)
        if controller.get("run_id") == receipt_role["run_id"]:
            matching_controllers.append((controller_path, controller))
    if len(matching_controllers) == 1:
        controller_path, controller = matching_controllers[0]
        controller["run_manifest_sha256"] = receipt_role[
            "run_manifest_sha256"
        ]
        write_json(controller_path, controller)
        receipt_role["controller_completion_receipt_sha256"] = (
            sha256_file(controller_path)
        )


def _copy_export_campaign(
    fixture: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    shutil.copytree(fixture["campaign_root"], destination)
    run_roots = {
        role: destination / fixture["contract"]["roles"][role][
            "run_relative_path"
        ]
        for role in ("baseline", "candidate_a", "candidate_b")
    }
    return {
        **fixture,
        "campaign_root": destination,
        "run_roots": run_roots,
        "receipt": deepcopy(fixture["receipt"]),
        "receipt_path": destination
        / fixture["receipt_path"].relative_to(fixture["campaign_root"]),
    }


def _patch_export_fixture_identity(
    monkeypatch: pytest.MonkeyPatch,
    fixture: dict[str, Any],
) -> None:
    common = fixture["contract"]["common_identity"]
    monkeypatch.setattr(
        open04_export_module,
        "_BASELINE_PACKAGE_SHA256",
        common["baseline_package_sha256"],
        raising=False,
    )
    monkeypatch.setattr(
        open04_export_module,
        "_BASELINE_MANIFEST_SHA256",
        common["baseline_manifest_sha256"],
        raising=False,
    )
    monkeypatch.setattr(
        open04_export_module,
        "_BASELINE_ATTESTATION_SHA256",
        common["release_attestation_sha256"],
        raising=False,
    )
    monkeypatch.setattr(
        open04_export_module,
        "verify_scenario_run",
        lambda _root: {"status": "pass"},
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_gzip_csv(
    path: Path,
    columns: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_post_receipt(
    contract: dict[str, Any],
    contract_sha256: str,
) -> dict[str, Any]:
    common = {
        key: contract["common_identity"][key]
        for key in OPEN04_POST_COMMON_IDENTITY_KEYS
    }
    terminal = {
        "baseline": (100.0, 0.0),
        "candidate_a": (99.0, 1.0),
        "candidate_b": (101.0, -1.0),
    }
    roles: dict[str, dict[str, Any]] = {}
    for index, role in enumerate(
        ("baseline", "candidate_a", "candidate_b"), start=3
    ):
        declared = contract["roles"][role]
        cost, tdc = terminal[role]
        roles[role] = {
            "scenario_id": declared["scenario_id"],
            "scenario_sha256": declared["scenario_sha256"],
            "scenario_source_sha256": declared[
                "scenario_source_sha256"
            ],
            "run_scenario_copy_sha256": declared[
                "scenario_source_sha256"
            ],
            "source_run_canonical_match_status": "pass",
            "compiled_inputs_digest": declared[
                "compiled_inputs_digest"
            ],
            "canonical_issuance_mix_sha256": declared[
                "canonical_issuance_mix_sha256"
            ],
            "issuance_mix_file_sha256": declared[
                "issuance_mix_file_sha256"
            ],
            "curve_sidecar_sha256": declared["curve_sidecar_sha256"],
            "signed_10y_shock_bp": declared["signed_10y_shock_bp"],
            "compiled_curve_delta_digest": declared[
                "compiled_curve_delta_digest"
            ],
            "run_id": f"{role}-run-id",
            "run_manifest_sha256": str(index) * 64,
            "run_relative_path": declared["run_relative_path"],
            "host": "BZOT",
            "controller_completion_receipt_sha256": str(index + 1) * 64,
            "terminal_summary_sha256": str(index + 2) * 64,
            "source_output_manifest_sha256": str(index + 3) * 64,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "event_count": 10,
            "event_root_sha256": str(index + 4) * 64,
            "verification_status": "pass",
            "financing_cost_component_identity_status": "pass",
            "accounting_invariants_status": "pass",
            "deterministic_replay_status": "pass",
            "memory_watchdog_status": "pass",
            "terminal_cumulative_modeled_financing_cost_bil": cost,
            "terminal_cumulative_tdc_change_bil": tdc,
            "overall_new_issuance_wam_years": declared[
                "overall_new_issuance_wam_years"
            ],
            "economic_changed_paths": declared[
                "economic_changed_paths"
            ],
            "physical_changed_inputs": declared[
                "physical_changed_inputs"
            ],
            "fixed_input_comparison_status": "pass",
            "fixed_input_records_sha256": declared[
                "fixed_input_records_sha256"
            ],
            "curve_date_set_sha256": declared[
                "curve_date_set_sha256"
            ],
            "tenor_set_sha256": declared["tenor_set_sha256"],
        }
    full_fy = [
        {
            "fiscal_year": year,
            "candidate_a": roles["candidate_a"][
                "overall_new_issuance_wam_years"
            ],
            "baseline": roles["baseline"][
                "overall_new_issuance_wam_years"
            ],
            "candidate_b": roles["candidate_b"][
                "overall_new_issuance_wam_years"
            ],
            "status": "pass",
        }
        for year in range(2027, 2037)
    ]
    return {
        "schema_version": OPEN04_POST_RUN_RECEIPT_SCHEMA_VERSION,
        "status": "pass",
        "campaign_eligible": True,
        "promotion_status": "eligible",
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": contract_sha256,
        "campaign_root_identity": contract["campaign_root_identity"],
        "verification_order": ["baseline", "candidate_a", "candidate_b"],
        "common_identity": common,
        "roles": roles,
        "pair_gates": {
            "threshold_bil": 0.000001,
            "candidate_a_financing_cost_delta_bil": -1.0,
            "candidate_a_tdc_delta_bil": 1.0,
            "candidate_b_financing_cost_delta_bil": 1.0,
            "candidate_b_tdc_delta_bil": -1.0,
            "terminal_sign_gate_status": "pass",
            "overall_wam_ordering": {
                "candidate_a": roles["candidate_a"][
                    "overall_new_issuance_wam_years"
                ],
                "baseline": roles["baseline"][
                    "overall_new_issuance_wam_years"
                ],
                "candidate_b": roles["candidate_b"][
                    "overall_new_issuance_wam_years"
                ],
                "status": "pass",
            },
            "full_fy_wam_ordering": full_fy,
            "full_fy_wam_gate_status": "pass",
            "controller_output_isolation_status": "pass",
            "aggregate_memory_budget_status": "pass",
            "evaluated_delta_antisymmetry_sha256": "f" * 64,
            "maximum_abs_evaluated_delta_antisymmetry_error": 0.0,
            "evaluated_delta_antisymmetry_status": "pass",
            "curve_date_set_equality_status": "pass",
            "tenor_set_equality_status": "pass",
            "change_perimeter_status": "pass",
            "fixed_input_equality_status": "pass",
        },
        "no_retuning_status": "pass",
    }


def _code_identity() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in open04_campaign_module._CODE_IDENTITY_KEYS:
        if key == "code_commit_sha":
            result[key] = "1" * 40
        elif key == "dirty_state":
            result[key] = False
        elif key == "python_version":
            result[key] = "3.14.5"
        elif key == "package_name":
            result[key] = "tdcsim"
        elif key == "package_version":
            result[key] = "0.1.0"
        elif key == "runtime_identity_source":
            result[key] = "installed_distribution_files"
        elif key == "requirements_lock_sha256":
            result[key] = (
                "16e3fc32257a01e1fd2e5a53867cc9e73496b9d35c6a5d4db19c671617325a4e"
            )
        elif key == "open04_exporter_source_sha256":
            result[key] = sha256_file(Path(open04_export_module.__file__))
        else:
            result[key] = "2" * 64
    result["dependency_lock_set_sha256"] = canonical_json_sha256(
        [
            {
                "relative_path": "uv.lock",
                "sha256": result["uv_lock_sha256"],
                "bytes": 10,
            },
            {
                "relative_path": "requirements.lock.txt",
                "sha256": result["requirements_lock_sha256"],
                "bytes": 20,
            },
        ]
    )
    return result


def _issuance_wam_years(mix: dict[str, Any]) -> float:
    fixed = mix["fixed_remainder_shares"]
    distributions = mix["maturity_distributions"]
    total = (
        float(mix["tips_share"])
        * _weighted_maturity(distributions["tips"])
        + float(mix["frn_share"])
        * _weighted_maturity(distributions["frn"])
    )
    fixed_remainder = 1.0 - float(mix["tips_share"]) - float(
        mix["frn_share"]
    )
    for security_type in ("bills", "notes", "bonds"):
        total += (
            fixed_remainder
            * float(fixed[security_type])
            * _weighted_maturity(distributions[security_type])
        )
    return total


def _weighted_maturity(rows: list[dict[str, float]]) -> float:
    return sum(
        float(row["maturity_years"]) * float(row["share"])
        for row in rows
    )


def _marked_baseline() -> dict[str, Any]:
    return {
        "schema_version": "tdcsim_cbo_scenario_v1",
        "scenario_id": OPEN04_ROLE_TO_SCENARIO_ID["baseline"],
        "baseline": dict(open04_campaign_module.OPEN04_BASELINE_IDENTITY),
        "provenance": {
            "kind": "user_stress_assumption",
            "label": "common_release_bound_baseline_noop",
        },
        "simulation": {
            "frequency": "daily",
            "start_date": _START,
            "end_date": _END,
        },
        "open04_campaign": {
            "contract_id": OPEN04_CAMPAIGN_CONTRACT_ID,
            "role": "baseline",
            "funding_closure_mode": OPEN04_FUNDING_CLOSURE_MODE,
        },
        "coupling": dict(OPEN04_FIXED_COUPLING),
        "overrides": {},
        "output": dict(OPEN04_OUTPUT_CONTRACT),
    }


def _pin_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "1")
