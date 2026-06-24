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
from tdcsim_cbo._json import sha256_file, write_json
from tdcsim_cbo.output import write_scenario_outputs
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
            "output": {"profile": "compact", "compression": "none", "include_final_portfolio": True},
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
