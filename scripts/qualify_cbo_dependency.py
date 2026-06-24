#!/usr/bin/env python3
"""Smoke-qualify a built TDCSIM wheel as a CBO scenario dependency."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


SCENARIO_SNIPPET = r"""
import json
import sys
from pathlib import Path

from tdcsim_cbo import CboBaselinePackage

baseline = CboBaselinePackage.open(sys.argv[1], attestation_path=sys.argv[2])
scenario = {
    "schema_version": "tdcsim_cbo_scenario_v1",
    "scenario_id": "dependency_noop_v1",
    "baseline": {
        "package_id": baseline.package_id,
        "package_sha256": baseline.package_sha256,
        "manifest_sha256": baseline.manifest_sha256,
        "release_attestation_sha256": baseline.attestation.sha256,
    },
    "provenance": {"kind": "tdcsim_transform_of_cbo_baseline", "label": "dependency qualification noop"},
    "coupling": {
        "frn_benchmark": "independent_explicit_path",
        "tips_real_yield": "recompute_from_nominal_and_scenario_inflation",
        "operating_cash_inflation": "baseline_cpi",
        "primary_deficit_to_debt_target": "independent_no_plug",
    },
    "overrides": {},
    "output": {"profile": "summary", "compression": "gzip"},
}
Path(sys.argv[3]).write_text(json.dumps(scenario, indent=2, sort_keys=True) + "\n", encoding="utf-8")
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("output/cbo_forecast_release_bound_package.zip"))
    parser.add_argument("--attestation", type=Path, default=Path("output/cbo_forecast_release_bound_attestation.json"))
    parser.add_argument("--run-scenario", action="store_true", help="also execute tdcsim-cbo run; can be slow on full packages")
    args = parser.parse_args(argv)
    wheel = args.wheel.resolve()
    if not wheel.exists():
        raise FileNotFoundError(wheel)
    baseline = args.baseline.resolve()
    attestation = args.attestation.resolve()
    with tempfile.TemporaryDirectory(prefix="tdcsim-cbo-qualify-") as tmp:
        tmp_path = Path(tmp)
        env = tmp_path / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
        pip = env / "bin" / "pip"
        python = env / "bin" / "python"
        cli = env / "bin" / "tdcsim-cbo"
        subprocess.run([str(pip), "install", "-q", str(wheel)], check=True)
        subprocess.run(
            [
                str(python),
                "-c",
                "from tdcsim_cbo import CboBaselinePackage, CboScenarioSpec, CboScenarioCompiler, run_cbo_scenario",
            ],
            check=True,
        )
        subprocess.run([str(cli), "--help"], check=True, stdout=subprocess.DEVNULL)
        if baseline.exists() and attestation.exists():
            scenario = tmp_path / "dependency_noop_scenario.json"
            compiled_root = tmp_path / "compiled"
            subprocess.run([str(python), "-c", SCENARIO_SNIPPET, str(baseline), str(attestation), str(scenario)], check=True)
            subprocess.run(
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
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
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
                check=True,
                stdout=subprocess.DEVNULL,
            )
            subprocess.run(
                [str(cli), "verify", "--compiled-dir", str(compiled_root / "compiled")],
                check=True,
                stdout=subprocess.DEVNULL,
            )
            if args.run_scenario:
                subprocess.run(
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
                        str(tmp_path / "run"),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
    print("pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
