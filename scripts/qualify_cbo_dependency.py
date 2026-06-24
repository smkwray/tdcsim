#!/usr/bin/env python3
"""Smoke-qualify a built TDCSIM wheel as a CBO scenario dependency."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


SCENARIO_SNIPPET = r"""
import csv
import hashlib
import json
import sys
from pathlib import Path

from tdcsim_cbo import CboBaselinePackage

baseline = CboBaselinePackage.open(sys.argv[1], attestation_path=sys.argv[2])
out = Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
materialized = baseline.materialize(out / "baseline")

def rows(path):
    import csv
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

primary = rows(materialized / "forecast_inputs" / "tdcsim_primary_deficit_path.csv")
debt = rows(materialized / "forecast_inputs" / "tdcsim_debt_stock_path.csv")
cash_residual = rows(materialized / "forecast_inputs" / "tdcsim_cash_reconciliation_residual.csv")
if primary and primary[0].get("period_start") and primary[0].get("period_end"):
    start_date = primary[0]["period_start"]
    end_date = primary[min(9, len(primary) - 1)]["period_end"]
else:
    dates = [row["period_end"] for row in debt if row.get("period_end")]
    start_date = dates[0]
    end_date = dates[min(9, len(dates) - 1)]

baseline_block = {
        "package_id": baseline.package_id,
        "package_sha256": baseline.package_sha256,
        "manifest_sha256": baseline.manifest_sha256,
        "release_attestation_sha256": baseline.attestation.sha256,
}

base = {
    "schema_version": "tdcsim_cbo_scenario_v1",
    "baseline": baseline_block,
    "simulation": {"frequency": "daily", "start_date": start_date, "end_date": end_date},
    "coupling": {
        "frn_benchmark": "independent_explicit_path",
        "tips_real_yield": "recompute_from_nominal_and_scenario_inflation",
        "operating_cash_inflation": "baseline_cpi",
        "primary_deficit_to_debt_target": "independent_no_plug",
    },
    "output": {"profile": "summary", "compression": "gzip"},
}

noop = {
    **base,
    "scenario_id": "dependency_noop_v1",
    "provenance": {"kind": "tdcsim_transform_of_cbo_baseline", "label": "dependency qualification noop"},
    "overrides": {},
}
compound = {
    **base,
    "scenario_id": "dependency_compound_v1",
    "provenance": {"kind": "user_stress_assumption", "label": "dependency qualification compound scenario"},
    "coupling": {
        "frn_benchmark": "derive_from_scenario_nominal_curve",
        "tips_real_yield": "recompute_from_nominal_and_scenario_inflation",
        "operating_cash_inflation": "scenario_cpi",
        "primary_deficit_to_debt_target": "independent_no_plug",
    },
    "overrides": {
        "nominal_yield_curve": {"mode": "parallel_bp", "shock_bp": 25},
        "frn_benchmark": {"mode": "linked_to_nominal_curve", "spread_bp": 5},
        "inflation_cpi": {"mode": "annualized_inflation_shift_bp", "shock_bp": 25, "terminal_rule": "carry_last_scenario_growth"},
        "tips_real_yield": {"mode": "linked_recompute", "additional_parallel_bp": 0},
        "operating_cash": {"mode": "constant_real"},
        "cash_reconciliation": {"mode": "zero", "funding_effect": "none"},
        "primary_deficit": {"mode": "scale_path", "scale": 1.01},
        "fiscal_incidence": {
            "mode": "static_shares",
            "domestic_ultimate_share": 0.50,
            "rest_of_world_share": 0.25,
            "foreign_official_share": 0.25,
            "other_share": 0.0,
        },
        "fed_holdings": {"mode": "scale_path", "scale": 1.0},
        "holder_preferences": {
            "mode": "static_shares",
            "rows": [
                {
                    "security_type": security_type,
                    "shares": {
                        "Banks": 0.10,
                        "CB": 0.0,
                        "Foreign": 0.20,
                        "Private": 0.70,
                        "TrustFunds": 0.0,
                        "FedInternal": 0.0,
                    },
                }
                for security_type in ("bills", "notes", "bonds", "tips", "frn")
            ],
        },
        "issuance_mix": {
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
        },
    },
}

cash_path = out / "file_backed_cash_residual.csv"
with cash_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "schema_version",
            "scenario_id",
            "period_start",
            "period_end",
            "cash_reconciliation_residual_bil",
            "component_type",
            "affects_operating_cash",
            "affects_primary_deficit",
            "affects_net_interest",
            "affects_total_deficit",
            "affects_debt_target",
            "affects_issuance_size",
            "affects_tdc_fiscal_flow",
        ],
    )
    writer.writeheader()
    for row in cash_residual:
        writer.writerow(
            {
                "schema_version": "tdcsim_cash_reconciliation_residual_v1",
                "scenario_id": row.get("scenario_id", "baseline"),
                "period_start": row["period_start"],
                "period_end": row["period_end"],
                "cash_reconciliation_residual_bil": "1.25",
                "component_type": "dependency_qualification_file_backed",
                "affects_operating_cash": "True",
                "affects_primary_deficit": "False",
                "affects_net_interest": "False",
                "affects_total_deficit": "False",
                "affects_debt_target": "False",
                "affects_issuance_size": "False",
                "affects_tdc_fiscal_flow": "False",
            }
        )

file_backed = {
    **base,
    "scenario_id": "dependency_file_backed_v1",
    "provenance": {"kind": "user_stress_assumption", "label": "dependency qualification file-backed scenario"},
    "overrides": {
        "cash_reconciliation": {
            "mode": "explicit_path_file",
            "file": {
                "relative_path": cash_path.name,
                "sha256": hashlib.sha256(cash_path.read_bytes()).hexdigest(),
                "media_type": "text/csv",
            },
            "funding_effect": "none",
        }
    },
}

for name, payload in {"noop": noop, "compound": compound, "file_backed": file_backed}.items():
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
"""


NEGATIVE_VERIFIER_SNIPPET = r"""
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

from tdcsim_cbo._json import read_json, write_json
from tdcsim_cbo.output import hash_output_tree, write_scenario_outputs
from tdcsim_cbo.verifier import VerificationError, verify_scenario_run

root = Path(sys.argv[1])
baseline = sys.argv[2]
attestation = sys.argv[3]
source = root / "run-noop"

def expect_reject(name, mutate, pattern):
    target = root / f"negative-{name}"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)
    manifest_path = target / "tdcsim_cbo_run_manifest.json"
    manifest = read_json(manifest_path)
    mutate(target, manifest)
    write_json(manifest_path, manifest)
    try:
        verify_scenario_run(target, baseline_package=baseline, attestation=attestation)
    except VerificationError as exc:
        if pattern not in str(exc):
            raise AssertionError(f"{name} rejected with wrong error: {exc}") from exc
        return
    raise AssertionError(f"{name} was not rejected")

def fail_validation(_target, manifest):
    manifest["validation"]["status"] = "fail"
    manifest["validation"]["invariants"][0]["status"] = "fail"
    manifest["boundary_checks"]["cash_residual_affects_issuance_size"] = ["True"]
    manifest["boundary_checks"]["cash_residual_nonfunding_flags"]["affects_issuance_size"] = ["True"]

def forge_identity(_target, manifest):
    manifest["code_environment"]["package_name"] = "fake-tdcsim"
    manifest["code_environment"]["requirements_lock_sha256"] = "0" * 64
    manifest["code_environment"]["wheel_sha256"] = "f" * 64

def forge_claim_boundary(_target, manifest):
    manifest["claim_boundary"]["net_interest_role"] = "binding_budgetary_net_interest"
    manifest["unsupported_components"] = []

def forge_python_version(_target, manifest):
    manifest["code_environment"]["python_version"] = "0.0.0-fabricated"

def forge_wheel_env_match(target, manifest):
    fake_wheel = target / "runtime" / "fake.whl"
    fake_wheel.parent.mkdir(exist_ok=True)
    fake_wheel.write_bytes(b"not a wheel")
    import hashlib
    fake_sha = hashlib.sha256(fake_wheel.read_bytes()).hexdigest()
    manifest["code_environment"]["wheel_sha256"] = fake_sha
    manifest["code_environment"]["wheel_artifact"] = {
        "logical_name": "fake.whl",
        "relative_path": "runtime/fake.whl",
        "sha256": fake_sha,
        "bytes": fake_wheel.stat().st_size,
        "media_type": "application/zip",
    }
    os.environ["TDCSIM_CBO_WHEEL_SHA256"] = fake_sha

def forge_one_row_output(target, manifest):
    results_item = next(item for item in manifest["outputs"] if str(item["logical_name"]).startswith("results_"))
    result_path = target / results_item["relative_path"]
    compression = "gzip" if str(result_path).endswith(".gz") else None
    results = pd.read_csv(result_path, compression=compression).iloc[[0]].copy()
    output_manifest = write_scenario_outputs(
        results,
        pd.DataFrame(),
        target / "outputs",
        profile=str(manifest["output_manifest"].get("profile", "summary")),
        compression=str(manifest["output_manifest"].get("compression", "gzip")),
        catalog_sqlite="catalog_sqlite" in manifest["output_manifest"],
    )
    output_hashes = hash_output_tree(target / "outputs")
    manifest["output_manifest"] = output_manifest
    manifest["output_hashes"] = output_hashes
    manifest["outputs"] = [
        {
            "logical_name": item["path"],
            "relative_path": f"outputs/{item['path']}",
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "media_type": "application/json" if item["path"].endswith(".json") else "text/csv",
        }
        for item in output_hashes
    ]

def remove_summary_key(target, manifest):
    results_item = next(item for item in manifest["outputs"] if str(item["logical_name"]).startswith("results_"))
    result_path = target / results_item["relative_path"]
    compression = "gzip" if str(result_path).endswith(".gz") else None
    results = pd.read_csv(result_path, compression=compression)
    if compression == "gzip":
        with gzip.open(result_path, "wt", encoding="utf-8", newline="") as handle:
            results.to_csv(handle, index=False)
    else:
        results.to_csv(result_path, index=False)
    summary_path = target / "outputs" / "summary.json"
    summary = read_json(summary_path)
    summary.pop("CBOFedAuctionShare_max_abs", None)
    write_json(summary_path, summary)
    manifest["output_hashes"] = hash_output_tree(target / "outputs")
    for item in manifest["outputs"]:
        path = target / item["relative_path"]
        for record in manifest["output_hashes"]:
            if record["path"] == Path(item["relative_path"]).relative_to("outputs").as_posix():
                item["sha256"] = record["sha256"]
                item["bytes"] = record["bytes"]

expect_reject("validation-fail", fail_validation, "validation")
expect_reject("forged-identity", forge_identity, "code_environment")
expect_reject("claim-boundary", forge_claim_boundary, "claim_boundary")
expect_reject("python-version", forge_python_version, "python_version")
expect_reject("wheel-env-match", forge_wheel_env_match, "wheel artifact digest")
os.environ["TDCSIM_CBO_WHEEL_SHA256"] = sys.argv[4]
expect_reject("one-row-output", forge_one_row_output, "engine replay output hash mismatch")
expect_reject("missing-summary-key", remove_summary_key, "engine replay output hash mismatch")
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("output/cbo_forecast_release_bound_package.zip"))
    parser.add_argument("--attestation", type=Path, default=Path("output/cbo_forecast_release_bound_attestation.json"))
    parser.add_argument("--compile-only", action="store_true", help="skip run/verify execution after validating and compiling scenarios")
    parser.add_argument("--constraints", type=Path, default=Path("requirements.lock.txt"), help="exact dependency lock to install before the wheel")
    parser.add_argument("--evidence-dir", type=Path, help="persist installed-wheel qualification evidence instead of using a temporary directory")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve()
    if not wheel.exists():
        raise FileNotFoundError(wheel)
    wheel_sha256 = _sha256(wheel)
    baseline = args.baseline.resolve()
    attestation = args.attestation.resolve()
    constraints = args.constraints.resolve()
    if not constraints.exists():
        raise FileNotFoundError(constraints)
    code_commit = _git_output(["rev-parse", "HEAD"])
    dirty_status = _git_output(["status", "--short"])
    if dirty_status:
        raise RuntimeError("CBO dependency qualification requires a clean git worktree")
    with _qualification_root(args.evidence_dir) as tmp_path:
        env = tmp_path / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        pip = env / "bin" / "pip"
        python = env / "bin" / "python"
        cli = env / "bin" / "tdcsim-cbo"
        run_env = os.environ.copy()
        run_env["TDCSIM_CBO_WHEEL_SHA256"] = wheel_sha256
        run_env["TDCSIM_CBO_WHEEL_PATH"] = str(wheel)
        run_env["TDCSIM_CBO_CODE_COMMIT_SHA"] = code_commit
        run_env["TDCSIM_CBO_DIRTY_STATE"] = "false"
        run_env["TDCSIM_CBO_REQUIREMENTS_LOCK_SHA256"] = _sha256(constraints)
        commands = []
        _run(commands, [str(pip), "install", "-q", "-r", str(constraints)], env=run_env)
        _run(commands, [str(pip), "install", "-q", "--no-deps", str(wheel)], env=run_env)
        _run(
            commands,
            [
                str(python),
                "-c",
                "import ratewall_paths; from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, CboScenarioCompiler, run_cbo_scenario",
            ],
            env=run_env,
        )
        _run(commands, [str(cli), "--help"], env=run_env, stdout=subprocess.DEVNULL)
        freeze = subprocess.run([str(pip), "freeze"], check=True, capture_output=True, text=True, env=run_env).stdout
        (tmp_path / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
        shutil.copyfile(wheel, tmp_path / wheel.name)
        shutil.copyfile(constraints, tmp_path / constraints.name)
        if not baseline.exists():
            raise FileNotFoundError(baseline)
        if not attestation.exists():
            raise FileNotFoundError(attestation)
        scenarios = tmp_path / "scenarios"
        _run(commands, [str(python), "-c", SCENARIO_SNIPPET, str(baseline), str(attestation), str(scenarios)], env=run_env)
        for name in ("noop", "compound", "file_backed"):
            scenario = scenarios / f"{name}.json"
            compiled_root = tmp_path / f"compiled-{name}"
            _run(
                commands,
                [
                    str(cli),
                    "validate",
                    "--baseline",
                    str(baseline),
                    "--attestation",
                    str(attestation),
                    "--scenario",
                    str(scenario),
                ],
                env=run_env,
                stdout=subprocess.DEVNULL,
            )
            _run(
                commands,
                [
                    str(cli),
                    "compile",
                    "--baseline",
                    str(baseline),
                    "--attestation",
                    str(attestation),
                    "--scenario",
                    str(scenario),
                    "--output-dir",
                    str(compiled_root),
                ],
                env=run_env,
                stdout=subprocess.DEVNULL,
            )
            _run(
                commands,
                [str(cli), "verify", "--compiled-dir", str(compiled_root / "compiled")],
                env=run_env,
                stdout=subprocess.DEVNULL,
            )
            if not args.compile_only:
                run_dir = tmp_path / f"run-{name}"
                _run(
                    commands,
                    [
                        str(cli),
                        "run",
                        "--baseline",
                        str(baseline),
                        "--attestation",
                        str(attestation),
                        "--scenario",
                        str(scenario),
                        "--output-dir",
                        str(run_dir),
                    ],
                    env=run_env,
                    stdout=subprocess.DEVNULL,
                )
                if name == "file_backed":
                    scenario.unlink()
                    (scenarios / "file_backed_cash_residual.csv").unlink()
                _run(
                    commands,
                    [
                        str(cli),
                        "verify",
                        "--run-dir",
                        str(run_dir),
                        "--baseline",
                        str(baseline),
                        "--attestation",
                        str(attestation),
                    ],
                    env=run_env,
                    stdout=subprocess.DEVNULL,
                )
        if not args.compile_only:
            _run(
                commands,
                [str(python), "-c", NEGATIVE_VERIFIER_SNIPPET, str(tmp_path), str(baseline), str(attestation), wheel_sha256],
                env=run_env,
            )
        freeze_sha256 = _sha256(tmp_path / "pip-freeze.txt")
        manifest = {
            "schema_version": "tdcsim_cbo_dependency_qualification_v1",
            "wheel": {"path": wheel.name, "sha256": wheel_sha256, "bytes": wheel.stat().st_size},
            "constraints": {"path": constraints.name, "sha256": _sha256(constraints), "bytes": constraints.stat().st_size},
            "baseline_package_sha256": _sha256(baseline),
            "attestation_sha256": _sha256(attestation),
            "pip_freeze_sha256": freeze_sha256,
            "artifact_hashes": _artifact_hashes(tmp_path, exclude_dirs={"venv"}),
            "compile_only": bool(args.compile_only),
            "commands": commands,
        }
        (tmp_path / "qualification_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("pass")
    return 0


@contextmanager
def _qualification_root(evidence_dir: Path | None):
    if evidence_dir is None:
        with tempfile.TemporaryDirectory(prefix="tdcsim-cbo-qualify-") as tmp:
            yield Path(tmp)
        return
    root = evidence_dir.expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"evidence directory must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    yield root


def _run(commands: list[dict], args: list[str], **kwargs) -> subprocess.CompletedProcess:
    completed = subprocess.run(args, check=True, **kwargs)
    commands.append({"command": args, "exit_code": completed.returncode})
    return completed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(args: list[str]) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _artifact_hashes(root: Path, *, exclude_dirs: set[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts):
            continue
        records.append({"path": rel.as_posix(), "sha256": _sha256(path), "bytes": path.stat().st_size})
    return records


if __name__ == "__main__":
    raise SystemExit(main())
