#!/usr/bin/env python3
"""Smoke-qualify a built TDCSIM wheel as a CBO scenario dependency."""

from __future__ import annotations

import argparse
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
for name, payload in {"noop": noop, "compound": compound}.items():
    (out / f"{name}.json").write_text(__import__("json").dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("output/cbo_forecast_release_bound_package.zip"))
    parser.add_argument("--attestation", type=Path, default=Path("output/cbo_forecast_release_bound_attestation.json"))
    parser.add_argument("--compile-only", action="store_true", help="skip run/verify execution after validating and compiling scenarios")
    parser.add_argument("--evidence-dir", type=Path, help="persist installed-wheel qualification evidence instead of using a temporary directory")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve()
    if not wheel.exists():
        raise FileNotFoundError(wheel)
    wheel_sha256 = _sha256(wheel)
    baseline = args.baseline.resolve()
    attestation = args.attestation.resolve()
    with _qualification_root(args.evidence_dir) as tmp_path:
        env = tmp_path / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        pip = env / "bin" / "pip"
        python = env / "bin" / "python"
        cli = env / "bin" / "tdcsim-cbo"
        run_env = os.environ.copy()
        run_env["TDCSIM_CBO_WHEEL_SHA256"] = wheel_sha256
        commands = []
        _run(commands, [str(pip), "install", "-q", str(wheel)], env=run_env)
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
        if not baseline.exists():
            raise FileNotFoundError(baseline)
        if not attestation.exists():
            raise FileNotFoundError(attestation)
        scenarios = tmp_path / "scenarios"
        _run(commands, [str(python), "-c", SCENARIO_SNIPPET, str(baseline), str(attestation), str(scenarios)], env=run_env)
        for name in ("noop", "compound"):
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
        manifest = {
            "schema_version": "tdcsim_cbo_dependency_qualification_v1",
            "wheel": {"path": wheel.name, "sha256": wheel_sha256, "bytes": wheel.stat().st_size},
            "baseline_package_sha256": _sha256(baseline),
            "attestation_sha256": _sha256(attestation),
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


if __name__ == "__main__":
    raise SystemExit(main())
