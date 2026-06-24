import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from forecast_paths import compiled_forecast_input_paths
from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, run_cbo_scenario
from tdcsim_cbo._json import read_json, sha256_file, write_json
from tdcsim_cbo.compiler import digest_input_tree, input_tree_hashes
from tdcsim_cbo.output import write_scenario_outputs
from tdcsim_cbo.runner import build_runtime_params
from tdcsim_cbo.verifier import VerificationError, verify_compiled_scenario, verify_scenario_run
from simulation_calendar import build_simulation_calendar
from test_cbo_engine_integration import (
    _build_temp_forecast_inputs,
    _dynamic_surface_path,
    _opening_controlled_portfolio,
    _write_cbo_fiscal_baseline,
    _write_frn_rate_path,
    _write_macro_path,
    _write_tips_forward_paths,
)
from test_tdcsim_cbo_baseline import RELEASE_SHA, VERIFIER_SHA


def test_compiled_forecast_input_paths_maps_required_engine_paths(tmp_path: Path) -> None:
    baseline, _ = _runner_baseline_and_scenarios(tmp_path)
    materialized = baseline.materialize(tmp_path / "materialized")

    paths = compiled_forecast_input_paths(materialized / "forecast_inputs")

    assert paths["debt_stock_path_file"].endswith("tdcsim_debt_stock_path.csv")
    assert paths["holder_absorption_path_file"].endswith("tdcsim_holder_profile_assumptions.csv")
    assert paths["tips_real_yield_path_file"].endswith("tdcsim_tips_real_yield_path.csv")


def test_run_cbo_scenario_writes_outputs_and_verifies(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)

    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run-noop")

    assert run.manifest_path.exists()
    assert run.results_path.exists()
    assert run.run_manifest["boundary_checks"]["net_interest_role"] == "diagnostic_nonbinding"
    assert run.run_manifest["boundary_checks"]["fed_target_holder_allocation_only"] is True
    assert verify_compiled_scenario(run.compiled.compiled_dir)["status"] == "pass"
    assert verify_scenario_run(run.output_dir)["status"] == "pass"
    assert verify_scenario_run(run.output_dir, baseline_package=baseline.package_path, attestation=baseline.attestation.path)["status"] == "pass"


def test_run_cbo_scenario_preserves_cash_non_sizing_boundary(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)

    noop = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run-noop")
    cash = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["cash"]), tmp_path / "run-cash")
    noop_results = pd.read_csv(noop.results_path)
    cash_results = pd.read_csv(cash.results_path)

    for col in ("CBORequiredFaceIssuance", "NewDebtIssued", "AuctionProceeds", "CBOControlledDebtPostIssuance"):
        assert cash_results[col].tolist() == pytest.approx(noop_results[col].tolist())
    assert cash_results["CBOCashReconciliationResidual"].sum() != pytest.approx(
        noop_results["CBOCashReconciliationResidual"].sum()
    )


def test_file_backed_run_package_is_self_contained_for_baseline_recompile(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)

    cash = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["cash"]), tmp_path / "run-cash")
    source_scenario = scenarios["cash"]
    source_override = tmp_path / "cash_residual_override.csv"
    source_scenario.unlink()
    source_override.unlink()

    scenario_block = cash.run_manifest["scenario"]
    assert scenario_block["referenced_files"][0]["relative_path"] == "cash_residual_override.csv"
    assert (cash.output_dir / "cash_residual_override.csv").exists()
    assert verify_scenario_run(
        cash.output_dir,
        baseline_package=baseline.package_path,
        attestation=baseline.attestation.path,
    )["status"] == "pass"


def test_run_cbo_scenario_net_interest_is_diagnostic_only(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)

    noop = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run-noop")
    net_interest = run_cbo_scenario(
        baseline,
        CboScenarioSpec.from_file(scenarios["net_interest"]),
        tmp_path / "run-net-interest",
    )
    noop_results = pd.read_csv(noop.results_path)
    ni_results = pd.read_csv(net_interest.results_path)

    for col in ("CBORequiredFaceIssuance", "NewDebtIssued", "AuctionProceeds", "CBOControlledDebtPostIssuance"):
        assert ni_results[col].tolist() == pytest.approx(noop_results[col].tolist())
    assert set(ni_results["NetInterestDiagnosticStatus"].dropna()) == {"cbo_reported_check_only"}


def test_run_cbo_scenario_fed_target_reallocates_only_and_remittance_unsupported(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)

    noop = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run-noop")
    fed = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["fed"]), tmp_path / "run-fed")
    noop_results = pd.read_csv(noop.results_path)
    fed_results = pd.read_csv(fed.results_path)

    assert fed_results["CBOFedAuctionShare"].abs().max() == pytest.approx(0.0)
    assert fed_results["CBORemittanceCashEffect"].fillna(0.0).abs().max() == pytest.approx(0.0)
    assert set(fed_results["CBORemittanceStatus"]) == {
        "not_modeled_cbo_primary_deficit_embeds_baseline_revenues"
    }
    assert fed_results["CB_Remittance"].isna().all()
    assert fed_results["CB_DeferredAsset"].isna().all()
    assert fed_results["CBORequiredFaceIssuance"].tolist() == pytest.approx(
        noop_results["CBORequiredFaceIssuance"].tolist()
    )


def test_output_gzip_bytes_are_deterministic(tmp_path: Path) -> None:
    results = pd.DataFrame({"Date": ["2027-01-01"], "CBORequiredFaceIssuance": [1.0]})
    portfolio = pd.DataFrame({"Status": ["Active"], "FaceValue": [1.0]})

    first = write_scenario_outputs(results, portfolio, tmp_path / "first", profile="compact", compression="gzip")
    second = write_scenario_outputs(results, portfolio, tmp_path / "second", profile="compact", compression="gzip")

    assert first["results"]["sha256"] == second["results"]["sha256"]
    assert first["final_portfolio"]["sha256"] == second["final_portfolio"]["sha256"]


def test_verifier_rejects_tampered_output_bytes(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    with run.results_path.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(VerificationError, match="output hash"):
        verify_scenario_run(run.output_dir)


def test_verifier_rejects_coordinated_manifest_and_output_tamper(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    results = pd.read_csv(run.results_path)
    results.loc[0, "CBOControlledDebtTargetError"] = 99.0
    results.to_csv(run.results_path, index=False)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    _refresh_result_hashes(run.output_dir, manifest)
    write_json(run.manifest_path, manifest)

    with pytest.raises(VerificationError, match="CBO target error"):
        verify_scenario_run(run.output_dir)


def test_verifier_rejects_missing_required_output_column_even_with_fresh_hashes(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    results = pd.read_csv(run.results_path).drop(columns=["CBOFedAuctionShare"])
    results.to_csv(run.results_path, index=False)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    _refresh_result_hashes(run.output_dir, manifest)
    write_json(run.manifest_path, manifest)

    with pytest.raises(VerificationError, match="missing required columns"):
        verify_scenario_run(run.output_dir)


def test_verifier_rejects_compiled_digest_split_even_with_updated_compiled_manifest(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    compiled_inputs = run.output_dir / "compile" / "compiled" / "forecast_inputs"
    target = compiled_inputs / "tdcsim_fiscal_incidence_policy.csv"
    rows = pd.read_csv(target)
    rows.loc[0, "du_share"] = 0.5
    rows.loc[0, "ru_share"] = 0.5
    rows.to_csv(target, index=False)
    compiled_manifest_path = run.output_dir / "compile" / "compiled" / "tdcsim_cbo_compiled_manifest.json"
    compiled_manifest = read_json(compiled_manifest_path)
    compiled_manifest["compiled_inputs_digest"] = digest_input_tree(compiled_inputs)
    compiled_manifest["input_hashes"] = input_tree_hashes(compiled_inputs)
    write_json(compiled_manifest_path, compiled_manifest)
    run_manifest = read_json(run.manifest_path)
    _refresh_compiled_input_hashes(run.output_dir, run_manifest)
    write_json(run.manifest_path, run_manifest)

    with pytest.raises(VerificationError, match="compiled input digest"):
        verify_scenario_run(run.output_dir)


def test_verifier_rejects_wrong_baseline_identity(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenarios["noop"]), tmp_path / "run")
    manifest = read_json(run.manifest_path)
    manifest["baseline"]["package_sha256"] = "0" * 64
    write_json(run.manifest_path, manifest)

    with pytest.raises(VerificationError, match="baseline package hash"):
        verify_scenario_run(run.output_dir, baseline_package=baseline.package_path, attestation=baseline.attestation.path)


def test_runtime_params_use_compiled_fiscal_incidence_policy(tmp_path: Path) -> None:
    baseline, _ = _runner_baseline_and_scenarios(tmp_path)
    scenario = _write_scenario(
        tmp_path / "fiscal-incidence.json",
        baseline,
        overrides={
            "fiscal_incidence": {
                "mode": "static_shares",
                "domestic_ultimate_share": 0.50,
                "rest_of_world_share": 0.25,
                "foreign_official_share": 0.25,
                "other_share": 0.0,
            }
        },
    )
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenario), tmp_path / "run-fiscal")

    policy = build_runtime_params(run.compiled.forecast_inputs_dir)["fiscal_incidence_policy"]

    assert policy["du_share"] == pytest.approx(0.50)
    assert policy["ru_share"] == pytest.approx(0.25)
    assert policy["foreign_share"] == pytest.approx(0.25)


def test_runtime_params_use_compiled_issuance_mix_artifact(tmp_path: Path) -> None:
    baseline, _ = _runner_baseline_and_scenarios(tmp_path)
    scenario = _write_scenario(
        tmp_path / "issuance-mix.json",
        baseline,
        overrides={"issuance_mix": _issuance_mix_override()},
    )
    run = run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenario), tmp_path / "run-issuance")

    params = build_runtime_params(run.compiled.forecast_inputs_dir)

    assert (run.compiled.forecast_inputs_dir / "tdcsim_issuance_mix_assumptions.json").exists()
    assert params["treasury_issuance_profile"]["TIPS"]["target_percentage"] == pytest.approx(0.08)
    assert params["treasury_issuance_profile"]["FRN"]["target_percentage"] == pytest.approx(0.04)
    assert params["funding_rule"]["negative_required_issuance_action"] == "retire_shortest_public_marketable"


def test_cb_auction_preferences_rejected_with_baseline_fed_target(tmp_path: Path) -> None:
    baseline, _ = _runner_baseline_and_scenarios(tmp_path)
    scenario = _write_scenario(
        tmp_path / "cb-holder.json",
        baseline,
        overrides={"holder_preferences": {"mode": "static_shares", "rows": _holder_preference_rows(cb_share=0.10)}},
    )

    with pytest.raises(ValueError, match="CB auction share"):
        run_cbo_scenario(baseline, CboScenarioSpec.from_file(scenario), tmp_path / "run-cb-holder")


def test_cli_validate_compile_and_verify(tmp_path: Path) -> None:
    baseline, scenarios = _runner_baseline_and_scenarios(tmp_path)
    compile_dir = tmp_path / "cli-compile"
    validate_cmd = [
        sys.executable,
        "-m",
        "tdcsim_cbo.cli",
        "validate",
        "--baseline",
        str(baseline.package_path),
        "--attestation",
        str(baseline.attestation.path),
        "--scenario",
        str(scenarios["noop"]),
    ]
    assert subprocess.run(validate_cmd, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True).stdout.strip() == b"pass"
    compile_cmd = [
        sys.executable,
        "-m",
        "tdcsim_cbo.cli",
        "compile",
        "--baseline",
        str(baseline.package_path),
        "--attestation",
        str(baseline.attestation.path),
        "--scenario",
        str(scenarios["noop"]),
        "--output-dir",
        str(compile_dir),
    ]
    subprocess.run(compile_cmd, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True)
    verify_cmd = [
        sys.executable,
        "-m",
        "tdcsim_cbo.cli",
        "verify",
        "--compiled-dir",
        str(compile_dir / "compiled"),
    ]
    assert subprocess.run(verify_cmd, cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True).stdout.strip() == b"pass"


def _runner_baseline_and_scenarios(tmp_path: Path) -> tuple[CboBaselinePackage, dict[str, Path]]:
    package, attestation = _write_runner_package(tmp_path)
    baseline = CboBaselinePackage.open(package, attestation_path=attestation)
    cash_residual_override = _write_cash_residual_override(tmp_path / "cash_residual_override.csv")
    scenarios = {
        "noop": _write_scenario(tmp_path / "noop.json", baseline, overrides={}),
        "cash": _write_scenario(
            tmp_path / "cash.json",
            baseline,
            overrides={
                "cash_reconciliation": {
                    "mode": "explicit_path_file",
                    "file": {
                        "relative_path": cash_residual_override.name,
                        "sha256": sha256_file(cash_residual_override),
                        "media_type": "text/csv",
                    },
                    "funding_effect": "none",
                },
            },
        ),
        "net_interest": _write_scenario(
            tmp_path / "net-interest.json",
            baseline,
            overrides={
                "net_interest_comparator": {"mode": "official_cbo_baseline", "role": "diagnostic_nonbinding"}
            },
        ),
        "fed": _write_scenario(
            tmp_path / "fed.json",
            baseline,
            overrides={"fed_holdings": {"mode": "scale_path", "scale": 1.1}},
        ),
    }
    return baseline, scenarios


def _write_cash_residual_override(path: Path) -> Path:
    periods = build_simulation_calendar("2026-09-20", "2026-09-30", "daily")
    _write_csv(
        path,
        [
            {
                "schema_version": "tdcsim_cash_reconciliation_residual_v1",
                "scenario_id": "baseline",
                "period_start": period.period_start.isoformat(),
                "period_end": period.period_end.isoformat(),
                "cash_reconciliation_residual_bil": 7.5,
                "component_type": "explicit_test_residual",
                "affects_operating_cash": True,
                "affects_primary_deficit": False,
                "affects_net_interest": False,
                "affects_total_deficit": False,
                "affects_debt_target": False,
                "affects_issuance_size": False,
                "affects_tdc_fiscal_flow": False,
            }
            for period in periods
        ],
    )
    return path


def _write_scenario(path: Path, baseline: CboBaselinePackage, *, overrides: dict) -> Path:
    write_json(
        path,
        {
            "schema_version": "tdcsim_cbo_scenario_v1",
            "scenario_id": path.stem.replace("-", "_") + "_v1",
            "baseline": {
                "package_id": baseline.package_id,
                "package_sha256": baseline.package_sha256,
                "manifest_sha256": baseline.manifest_sha256,
                "release_attestation_sha256": baseline.attestation.sha256,
            },
            "provenance": {"kind": "user_stress_assumption", "label": "closeout fixture"},
            "simulation": {"frequency": "daily", "start_date": "2026-09-20", "end_date": "2026-09-30"},
            "coupling": {
                "frn_benchmark": "independent_explicit_path",
                "tips_real_yield": "recompute_from_nominal_and_scenario_inflation",
                "operating_cash_inflation": "baseline_cpi",
                "primary_deficit_to_debt_target": "independent_no_plug",
            },
            "overrides": overrides,
            "output": {"profile": "compact", "compression": "none"},
        },
    )
    return path


def _write_runner_package(tmp_path: Path) -> tuple[Path, Path]:
    package_dir = tmp_path / "runner_pkg"
    inputs = package_dir / "forecast_inputs"
    inputs.mkdir(parents=True)
    (package_dir / "requirements.lock.txt").write_text("pandas==0\n", encoding="utf-8")
    lock_sha = sha256_file(package_dir / "requirements.lock.txt")
    periods = build_simulation_calendar("2026-09-20", "2026-09-30", "daily")
    input_paths = _build_temp_forecast_inputs(
        tmp_path / "raw_inputs",
        periods=periods,
        period_start="2026-09-20",
        period_end="2026-09-30",
    )
    extra_paths = {
        "tdcsim_yield_curve_surface.csv": _dynamic_surface_path(tmp_path / "raw_inputs"),
        "tdcsim_frn_rate_path.csv": _write_frn_rate_path(tmp_path / "raw_inputs"),
        **{path.name: path for path in _write_tips_forward_paths(tmp_path / "raw_inputs").values()},
        "tdcsim_macro_forecast_path.csv": _write_macro_path(tmp_path / "raw_inputs"),
        "tdcsim_cbo_fiscal_baseline.csv": _write_cbo_fiscal_baseline(tmp_path / "raw_inputs"),
    }
    for path in input_paths.values():
        (inputs / path.name).write_bytes(path.read_bytes())
    for name, path in extra_paths.items():
        (inputs / name).write_bytes(path.read_bytes())
    _write_csv(inputs / "source_fixtures.csv", [{"schema_version": "fixture", "source_role": "diagnostic"}])
    _write_csv(inputs / "tdcsim_current_fy_splice.csv", [{"schema_version": "fixture", "scenario_id": "baseline"}])
    _write_csv(
        inputs / "tdcsim_holder_profile_assumptions.csv",
        [
            {"schema_version": "fixture", "scenario_id": "baseline", "holder_type": "Private", "holder_subbucket": "", "source_role": "scenario_assumption", "runtime_role": "memo_only", "claim_boundary": "fixture", "bills_pct": 1.0, "notes_pct": 1.0, "bonds_pct": 1.0, "tips_pct": 1.0, "frn_pct": 1.0},
            {"schema_version": "fixture", "scenario_id": "baseline", "holder_type": "CB", "holder_subbucket": "", "source_role": "scenario_assumption", "runtime_role": "memo_only", "claim_boundary": "fixture", "bills_pct": 0.0, "notes_pct": 0.0, "bonds_pct": 0.0, "tips_pct": 0.0, "frn_pct": 0.0},
        ],
    )
    _write_csv(
        inputs / "tdcsim_fed_holdings_path.csv",
        [
            {
                "schema_version": "tdcsim_fed_holdings_path_v1",
                "scenario_id": "baseline",
                "period_end": period.period_end.isoformat(),
                "holder_type": "CB",
                "cbo_fed_holdings_target_bil": 100.0,
                "interpolation_method": "fixture",
                "source_fiscal_year": 2026,
                "source_role": "scenario_assumption",
                "runtime_role": "hard_target",
                "observation_date": "2026-09-20",
                "available_date": "2026-09-20",
                "source_status": "fixture_fed_holdings_target",
                "claim_boundary": "fed_holdings_path_guides_holder_allocation_not_total_issuance",
            }
            for period in periods
        ],
    )
    _opening_controlled_portfolio(1_000.0).to_csv(inputs / "tdcsim_opening_portfolio.csv", index=False)
    write_json(inputs / "tdcsim_opening_portfolio_metadata.json", {"schema_version": "fixture"})
    _write_csv(inputs / "tdcsim_opening_frn_indexation_diagnostics.csv", [{"schema_version": "fixture"}])
    _write_csv(inputs / "tdcsim_opening_rollforward_diagnostics.csv", [{"schema_version": "fixture"}])
    _write_csv(inputs / "tdcsim_opening_tips_indexation_diagnostics.csv", [{"schema_version": "fixture"}])

    manifest = {
        "schema_version": "tdcsim_cbo_forecast_smoke_manifest_v1",
        "scenario_id": "baseline",
        "trust_anchor": {
            "schema_version": "tdcsim_cbo_package_trust_anchor_v1",
            "code_revision": RELEASE_SHA,
            "dirty_state": False,
            "requirements_lock_sha256": lock_sha,
            "verifier_sha256": VERIFIER_SHA,
        },
    }
    write_json(package_dir / "manifest.json", manifest)
    write_json(inputs / "source_contract_smoke.json", manifest)
    package = tmp_path / "runner_pkg.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir).as_posix())
    attestation = tmp_path / "runner_attestation.json"
    write_json(
        attestation,
        {
            "schema_version": "tdcsim_cbo_external_release_attestation_v1",
            "attestation_created_at_utc": "2026-06-24T00:00:00+00:00",
            "release_commit_sha": RELEASE_SHA,
            "release_commit_short": "639c033",
            "branch_state": "detached_release_validation_worktree",
            "worktree_path": "/tmp/tdcsim-cbo-release-639c033",
            "git_status_short": "",
            "dirty_state": False,
            "baseline_package_dir": "output/cbo_forecast_release_bound",
            "baseline_package_zip": "output/cbo_forecast_release_bound_package.zip",
            "baseline_package_zip_sha256": sha256_file(package),
            "baseline_manifest_sha256": sha256_file(package_dir / "manifest.json"),
            "source_contract_sha256": sha256_file(inputs / "source_contract_smoke.json"),
            "verifier_sha256": VERIFIER_SHA,
            "requirements_lock_sha256": lock_sha,
            "manifest_trust_anchor": manifest["trust_anchor"],
            "artifact_hashes_count": 0,
            "required_files_count": 20,
            "scenario_id": "baseline",
            "run_summaries": [],
            "source_package_used": "test-fixture",
            "validation_grade": "release_reproducible_local_clean_commit",
            "claim_boundary": {"package_role": "test"},
            "unsupported_components": {},
            "commands": [{"name": "unit_fixture", "command": "test", "cwd": str(tmp_path), "exit_code": 0, "stdout_log": "fixture.txt", "stdout_sha256": "0" * 64, "result": "pass"}],
            "attestation_log_hashes": {},
        },
    )
    return package, attestation


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _refresh_result_hashes(run_dir: Path, manifest: dict) -> None:
    result_item = next(item for item in manifest["outputs"] if str(item["logical_name"]).startswith("results_"))
    result_rel = result_item["relative_path"]
    result_path = run_dir / result_rel
    result_record = {
        "logical_name": result_path.name,
        "relative_path": result_rel,
        "sha256": sha256_file(result_path),
        "bytes": result_path.stat().st_size,
        "media_type": "text/csv",
    }
    for index, item in enumerate(manifest["outputs"]):
        if item["relative_path"] == result_rel:
            manifest["outputs"][index] = result_record
    for index, item in enumerate(manifest["output_hashes"]):
        if item["path"] == Path(result_rel).name:
            manifest["output_hashes"][index] = {
                "path": Path(result_rel).name,
                "bytes": result_path.stat().st_size,
                "sha256": sha256_file(result_path),
            }


def _refresh_compiled_input_hashes(run_dir: Path, manifest: dict) -> None:
    compiled_inputs = run_dir / "compile" / "compiled" / "forecast_inputs"
    by_path = {
        f"compile/compiled/forecast_inputs/{item['path']}": item
        for item in input_tree_hashes(compiled_inputs)
    }
    for item in manifest["compiled_inputs"]:
        updated = by_path[item["relative_path"]]
        item["sha256"] = updated["sha256"]
        item["bytes"] = updated["bytes"]


def _holder_preference_rows(*, cb_share: float = 0.0) -> list[dict]:
    private_share = 1.0 - cb_share
    return [
        {
            "security_type": security_type,
            "shares": {
                "Banks": 0.0,
                "CB": cb_share,
                "Foreign": 0.0,
                "Private": private_share,
                "TrustFunds": 0.0,
                "FedInternal": 0.0,
            },
        }
        for security_type in ("bills", "notes", "bonds", "tips", "frn")
    ]


def _issuance_mix_override() -> dict:
    return {
        "mode": "replace_shares",
        "tips_share": 0.08,
        "frn_share": 0.04,
        "fixed_remainder_shares": {"bills": 0.30, "notes": 0.50, "bonds": 0.20},
        "maturity_distributions": {
            "bills": [{"maturity_years": 0.5, "share": 1.0}],
            "notes": [{"maturity_years": 5.0, "share": 1.0}],
            "bonds": [{"maturity_years": 20.0, "share": 1.0}],
            "tips": [{"maturity_years": 10.0, "share": 1.0}],
            "frn": [{"maturity_years": 2.0, "share": 1.0}],
        },
        "negative_issuance_action": "retire_shortest_public_marketable",
    }
