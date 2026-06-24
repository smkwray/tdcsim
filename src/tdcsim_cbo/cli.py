"""Command-line interface for TDCSIM CBO baseline scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .baseline import CboBaselinePackage
from .compiler import CboScenarioCompiler
from .contract import CboScenarioSpec
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

    args = parser.parse_args(argv)
    if args.command == "verify":
        if bool(args.compiled_dir) == bool(args.run_dir):
            parser.error("verify requires exactly one of --compiled-dir or --run-dir")
        result = verify_compiled_scenario(args.compiled_dir) if args.compiled_dir else verify_scenario_run(args.run_dir)
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
        print(run.run_manifest["outputs"]["summary"]["sha256"])
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)


if __name__ == "__main__":
    raise SystemExit(main())
