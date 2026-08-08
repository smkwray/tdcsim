"""Command-line interface for TDCSIM CBO baseline scenarios."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ._json import sha256_bytes, sha256_file
from .baseline import CboBaselinePackage
from .compiler import CboScenarioCompiler
from .contract import CboScenarioSpec
from .forecast_state import (
    export_forecast_state_package,
    forecast_state_window,
    run_no_shock_rollforward,
)
from .marginal_tdc import assemble_marginal_tdc_pair, verify_marginal_tdc_pair
from .process_watchdog import (
    THREAD_LIMIT_ENVIRONMENT_VARIABLES,
    run_command_with_watchdog,
)
from .runner import (
    cleanup_watchdog_intervention,
    consume_watchdog_failure_handoff,
    finalize_watchdog_handoff,
    run_cbo_scenario,
)
from .verifier import verify_compiled_scenario, verify_scenario_run


_PYVENV_LAUNCHER_ENV = "__PYVENV_LAUNCHER__"


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
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
    run_cmd.add_argument("--_watchdog-worker", action="store_true", help=argparse.SUPPRESS)
    run_cmd.add_argument("--_watchdog-handoff", help=argparse.SUPPRESS)

    verify = sub.add_parser("verify", help="Verify compiled or run outputs")
    verify.add_argument("--compiled-dir")
    verify.add_argument("--run-dir")
    verify.add_argument("--baseline", type=Path)
    verify.add_argument("--attestation", type=Path)

    marginal_pair = sub.add_parser("assemble-marginal-pair", help="Assemble a RateWall marginal TDC pair")
    marginal_pair.add_argument("--baseline", type=Path)
    marginal_pair.add_argument("--attestation", type=Path)
    marginal_pair.add_argument("--pair-spec", required=True, type=Path)
    marginal_pair.add_argument("--output-dir", required=True, type=Path)
    marginal_pair.add_argument("--source-run-catalog", type=Path)

    verify_marginal_pair = sub.add_parser("verify-marginal-pair", help="Verify a RateWall marginal TDC pair")
    verify_marginal_pair.add_argument("--baseline", type=Path)
    verify_marginal_pair.add_argument("--attestation", type=Path)
    verify_marginal_pair.add_argument("--pair-dir", required=True, type=Path)
    verify_marginal_pair.add_argument("--source-run-catalog", type=Path)

    export_forecast = sub.add_parser("export-forecast-state", help="Export a derived forecast opening-state package")
    _add_baseline_args(export_forecast)
    export_forecast.add_argument("--year", required=True, type=int)
    export_forecast.add_argument("--output-dir", required=True, type=Path)
    export_forecast.add_argument("--work-dir", required=True, type=Path)
    export_forecast.add_argument("--force", action="store_true")

    args = parser.parse_args(raw_argv)
    if (
        args.command == "run"
        and args._watchdog_worker
        and not args._watchdog_handoff
    ):
        parser.error(
            "the watchdog worker requires its paired parent handoff"
        )
    if args.command == "run" and not args._watchdog_worker:
        output_dir = Path(args.output_dir).expanduser().resolve()
        handoff_path = output_dir.with_name(
            f".{output_dir.name}.watchdog-handoff-{uuid4().hex}.json"
        )
        worker_log_path = handoff_path.with_name(
            f"{handoff_path.stem}.worker.log"
        )
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        worker_executable, venv_launcher = _watchdog_worker_launch()
        worker_command = [
            worker_executable,
            "-m",
            "tdcsim_cbo.cli",
            *raw_argv,
            "--_watchdog-worker",
            "--_watchdog-handoff",
            str(handoff_path),
        ]

        def complete_parent_acceptance(result) -> bool:
            manifest = finalize_watchdog_handoff(
                output_dir,
                handoff_path,
                result,
            )
            if manifest is None:
                return False
            _emit_watchdog_controller_message(
                manifest["output_manifest"]["summary"]["sha256"]
            )
            return True

        previous_thread_environment = _pin_numerical_threads_to_one()
        launcher_was_present = _PYVENV_LAUNCHER_ENV in os.environ
        previous_launcher = os.environ.get(_PYVENV_LAUNCHER_ENV)
        if venv_launcher is not None:
            os.environ[_PYVENV_LAUNCHER_ENV] = venv_launcher
        try:
            with worker_log_path.open("x+b", buffering=0) as worker_log:
                def worker_log_evidence() -> dict[str, object]:
                    worker_log.flush()
                    end = worker_log.seek(0, os.SEEK_END)
                    worker_log.seek(0)
                    digest = sha256_bytes(worker_log.read())
                    worker_log.seek(end)
                    return {
                        "file_name": worker_log_path.name,
                        "bytes": end,
                        "sha256": digest,
                    }

                def start_worker(command: Sequence[str]):
                    return subprocess.Popen(
                        list(command),
                        stdout=worker_log,
                        stderr=subprocess.STDOUT,
                    )

                sample_count = 0

                def report_sample(
                    child_pid: int,
                    last_rss_bytes: int,
                    peak_rss_bytes: int,
                ) -> None:
                    nonlocal sample_count
                    sample_count += 1
                    if sample_count != 1 and sample_count % 30:
                        return
                    _emit_watchdog_controller_message(
                        "tdcsim: watchdog_checkpoint "
                        + json.dumps(
                            {
                                "child_pid": child_pid,
                                "last_rss_bytes": last_rss_bytes,
                                "peak_rss_bytes": peak_rss_bytes,
                                "sample_count": sample_count,
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                    )

                def failure_details(result) -> dict[str, object]:
                    try:
                        details = dict(
                            consume_watchdog_failure_handoff(
                                output_dir,
                                handoff_path,
                                result,
                            )
                        )
                    except Exception as exc:
                        details = {
                            "watchdog_handoff_status": "unavailable",
                            "watchdog_handoff_error": type(exc).__name__,
                        }
                    details["worker_standard_stream_log"] = (
                        worker_log_evidence()
                    )
                    return details

                def intervention_details(result) -> dict[str, object]:
                    details = dict(
                        cleanup_watchdog_intervention(
                            output_dir,
                            handoff_path,
                            result,
                        )
                    )
                    details["worker_standard_stream_log"] = (
                        worker_log_evidence()
                    )
                    return details

                status = run_command_with_watchdog(
                    worker_command,
                    output_dir=output_dir,
                    popen_factory=start_worker,
                    sample_callback=report_sample,
                    completion_callback=complete_parent_acceptance,
                    failure_details_callback=failure_details,
                    intervention_details_callback=intervention_details,
                )
            if status == 0:
                try:
                    worker_log_path.unlink()
                except OSError:
                    pass
            return status
        finally:
            _restore_thread_environment(previous_thread_environment)
            if venv_launcher is not None:
                if launcher_was_present:
                    assert previous_launcher is not None
                    os.environ[_PYVENV_LAUNCHER_ENV] = previous_launcher
                else:
                    os.environ.pop(_PYVENV_LAUNCHER_ENV, None)
    if args.command == "assemble-marginal-pair":
        result = assemble_marginal_tdc_pair(
            args.pair_spec,
            args.output_dir,
            baseline_package=args.baseline,
            attestation=args.attestation,
            source_run_catalog=args.source_run_catalog,
        )
        print(result.manifest_path)
        return 0
    if args.command == "verify-marginal-pair":
        result = verify_marginal_tdc_pair(
            args.pair_dir,
            baseline_package=args.baseline,
            attestation=args.attestation,
            source_run_catalog=args.source_run_catalog,
        )
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
        run = run_cbo_scenario(
            baseline,
            spec,
            args.output_dir,
            output_profile=args.profile,
            watchdog_handoff=args._watchdog_handoff,
        )
        if args._watchdog_handoff is None:
            print(run.run_manifest["output_manifest"]["summary"]["sha256"])
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


def _add_baseline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)


def _pin_numerical_threads_to_one() -> dict[str, str | None]:
    previous = {
        name: os.environ.get(name)
        for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    }
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        os.environ[name] = "1"
    return previous


def _watchdog_worker_launch() -> tuple[str, str | None]:
    """Return a direct worker executable and optional Windows venv identity.

    CPython's Windows virtual-environment executable is a redirector that
    creates and waits on the real interpreter. Watching that redirector would
    miss the simulator's RSS and make its PID differ from the worker-owned
    claim and handoff. Launching the base interpreter with CPython's own
    ``__PYVENV_LAUNCHER__`` identity preserves the venv while making the
    watched process the actual interpreter.
    """

    if platform.system() != "Windows" or sys.prefix == sys.base_prefix:
        return sys.executable, None
    base_value = str(getattr(sys, "_base_executable", "") or "")
    if not base_value:
        raise RuntimeError(
            "Windows venv watchdog launch requires sys._base_executable"
        )
    base_executable = Path(base_value).expanduser().resolve()
    if not base_executable.is_file():
        raise RuntimeError(
            "Windows venv watchdog base interpreter is missing: "
            f"{base_executable}"
        )
    if os.path.normcase(str(base_executable)) == os.path.normcase(
        str(Path(sys.executable).expanduser().resolve())
    ):
        raise RuntimeError(
            "Windows venv watchdog base interpreter resolves to the "
            "redirector executable"
        )
    return str(base_executable), sys.executable


def _emit_watchdog_controller_message(message: object) -> None:
    """Keep controller output live without making a lost stream kill a run."""

    try:
        print(message, flush=True)
    except (AttributeError, OSError, ValueError):
        try:
            stream_fd = sys.stdout.fileno()
            null_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(null_fd, stream_fd)
            finally:
                os.close(null_fd)
        except (AttributeError, OSError, ValueError):
            pass


def _restore_thread_environment(previous: dict[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
