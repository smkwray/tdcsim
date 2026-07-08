"""Command-line interface for TDCSIM CBO baseline scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .baseline import CboBaselinePackage
from .compiler import CboScenarioCompiler
from .contract import CboScenarioSpec
from .forecast_state import (
    export_forecast_state_package,
    forecast_state_window,
    run_no_shock_rollforward,
)
from .marginal_tdc import assemble_marginal_tdc_pair, verify_marginal_tdc_pair
from .runner import run_cbo_scenario
from .verifier import verify_compiled_scenario, verify_scenario_run


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tdcsim-cbo")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate a baseline package and scenario spec")
    _add_baseline_args(validate)
    validate.add_argument("--scenario", required=True)

    compile_cmd = sub.add_parser("compile", help="Compile scenario forecast inputs")
    _add_baseline_args(compile_cmd)
    compile_cmd.add_argument("--scenario", required=True)
    compile_cmd.add_argument("--output-dir", required=True)

    run_cmd = sub.add_parser("run", help="Compile and run a CBO scenario")
    _add_baseline_args(run_cmd)
    run_cmd.add_argument("--scenario", required=True)
    run_cmd.add_argument("--output-dir", required=True)
    run_cmd.add_argument("--profile", choices=["summary", "compact", "audit"], default=None)

    verify = sub.add_parser("verify", help="Verify compiled or run outputs")
    verify.add_argument("--compiled-dir")
    verify.add_argument("--run-dir")
    verify.add_argument("--baseline", type=Path)
    verify.add_argument("--attestation", type=Path)

    marginal_pair = sub.add_parser("assemble-marginal-pair", help="Assemble a RateWall marginal TDC pair")
    marginal_pair.add_argument("--pair-spec", required=True, type=Path)
    marginal_pair.add_argument("--output-dir", required=True, type=Path)

    verify_marginal_pair = sub.add_parser("verify-marginal-pair", help="Verify a RateWall marginal TDC pair")
    verify_marginal_pair.add_argument("--pair-dir", required=True, type=Path)

    export_forecast = sub.add_parser("export-forecast-state", help="Export a derived forecast opening-state package")
    _add_baseline_args(export_forecast)
    export_forecast.add_argument("--year", required=True, type=int)
    export_forecast.add_argument("--output-dir", required=True, type=Path)
    export_forecast.add_argument("--work-dir", required=True, type=Path)
    export_forecast.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "assemble-marginal-pair":
        result = assemble_marginal_tdc_pair(args.pair_spec, args.output_dir)
        print(result.manifest_path)
        return 0
    if args.command == "verify-marginal-pair":
        result = verify_marginal_tdc_pair(args.pair_dir)
        print(result["status"])
        return 0
    if args.command == "export-forecast-state":
        baseline = CboBaselinePackage.open(args.baseline, attestation_path=args.attestation)
        window = forecast_state_window(args.year)
        rollforward = run_no_shock_rollforward(
            baseline,
            state_window=window,
            output_dir=args.work_dir / "rollforward",
            scenario_dir=args.work_dir / "scenarios",
            force=args.force,
        )
        result = export_forecast_state_package(
            baseline,
            state_window=window,
            rollforward_run_dir=rollforward,
            output_zip=args.output_dir / f"cbo_baseline_state_{args.year}_opening_package.zip",
            output_attestation=args.output_dir / f"cbo_baseline_state_{args.year}_opening_attestation.json",
            output_manifest=args.output_dir / f"cbo_baseline_state_{args.year}_opening_export_manifest.json",
            force=args.force,
        )
        print(result.package_zip)
        return 0
    if args.command == "verify":
        if bool(args.compiled_dir) == bool(args.run_dir):
            parser.error("verify requires exactly one of --compiled-dir or --run-dir")
        result = (
            verify_compiled_scenario(args.compiled_dir)
            if args.compiled_dir
            else verify_scenario_run(
                args.run_dir,
                baseline_package=args.baseline,
                attestation=args.attestation,
            )
        )
        print(result["status"])
        return 0

    baseline = CboBaselinePackage.open(args.baseline, attestation_path=args.attestation)
    spec = CboScenarioSpec.from_file(args.scenario)
    spec.assert_baseline_matches(baseline)
    if args.command == "validate":
        print("pass")
        return 0
    if args.command == "compile":
        compiled = CboScenarioCompiler().compile(baseline, spec, args.output_dir)
        print(compiled.compiled_inputs_digest)
        return 0
    if args.command == "run":
        run = run_cbo_scenario(baseline, spec, args.output_dir, output_profile=args.profile)
        print(run.run_manifest["output_manifest"]["summary"]["sha256"])
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)


if __name__ == "__main__":
    raise SystemExit(main())
