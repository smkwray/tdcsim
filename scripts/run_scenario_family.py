#!/usr/bin/env python3
"""Run, verify and export an illustrative CBO scenario family from the installed release.

``run`` launches the family's scenarios in parallel through the CBO CLI with
opt-in TRC routing; ``verify`` replays each completed run with the bounded
verifier; ``export`` writes the fiscal-year family table and receipt. ``run``
and ``verify`` accept ``--roles`` so that rate-path scenarios can follow the
quantity runs they are derived from. Each stage fails loudly and leaves earlier
stage outputs untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_open04_reproduction import (  # noqa: E402
    _assert_installed_distribution_import,
    _fail,
)


def _family(args: argparse.Namespace) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    family = json.loads(Path(args.family_spec).read_text(encoding="utf-8"))
    scenario_dir = Path(args.scenario_dir)
    roles = [item["role"] for item in family["scenarios"]]
    selected = getattr(args, "roles", None)
    selected = selected.split(",") if selected else roles
    unknown = sorted(set(selected) - set(roles))
    if unknown:
        _fail(f"unknown roles: {unknown}")
    specs = []
    for index, item in enumerate(family["scenarios"]):
        if item["role"] not in selected:
            continue
        path = scenario_dir / f"{index:02d}_{item['role']}.json"
        if not path.is_file():
            _fail(f"written scenario is missing: {path}")
        specs.append((str(item["role"]), path))
    return family, specs


def _thread_environment() -> dict[str, str]:
    from tdcsim_cbo.process_watchdog import THREAD_LIMIT_ENVIRONMENT_VARIABLES

    environment = dict(os.environ)
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        environment[name] = "1"
    return environment


def stage_run(args: argparse.Namespace) -> int:
    _assert_installed_distribution_import()
    _, specs = _family(args)
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    routing = str(Path(args.trc_routing).resolve())
    processes = {}
    for role, scenario in specs:
        # The bounded runner claims a run's parent directory, so each role gets its own.
        run_dir = root / role / "run"
        if run_dir.parent.exists():
            _fail(f"role directory already exists: {run_dir.parent}")
        run_dir.parent.mkdir(parents=True)
        command = [
            sys.executable, "-B", "-m", "tdcsim_cbo.cli", "run",
            "--baseline", str(Path(args.baseline).resolve()), "--attestation", str(Path(args.attestation).resolve()),
            "--scenario", str(scenario), "--output-dir", str(run_dir), "--trc-routing", routing,
        ]
        log = (root / role / "run.log").open("ab")
        processes[role] = (subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=_thread_environment(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        ), log)
        print(f"launched {role}: pid {processes[role][0].pid}", flush=True)
    failures = []
    for role, (process, log) in processes.items():
        code = process.wait()
        log.close()
        print(f"{role}: exit {code}", flush=True)
        if code != 0:
            failures.append(role)
    if failures:
        _fail(f"scenario runs failed: {failures}")
    return 0


def stage_verify_one(args: argparse.Namespace) -> int:
    from tdcsim_cbo.verifier import verify_scenario_run

    result = verify_scenario_run(
        args.run_dir, baseline_package=Path(args.baseline), attestation=Path(args.attestation)
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return 0 if result.get("status") == "pass" else 1


def stage_verify(args: argparse.Namespace) -> int:
    _assert_installed_distribution_import()
    _, specs = _family(args)
    root = Path(args.output_root).resolve()
    processes = {}
    for role, _scenario in specs:
        out = root / role / "verification.json"
        if out.exists():
            _fail(f"verification record already exists: {out}")
        command = [
            sys.executable, "-B", str(Path(__file__).resolve()), "verify-one",
            "--run-dir", str(root / role / "run"), "--out", str(out),
            "--baseline", str(Path(args.baseline).resolve()), "--attestation", str(Path(args.attestation).resolve()),
        ]
        log = (root / role / "verify.log").open("ab")
        processes[role] = (subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=_thread_environment(), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        ), log)
    failures = []
    for role, (process, log) in processes.items():
        code = process.wait()
        log.close()
        print(f"{role}: verify exit {code}", flush=True)
        if code != 0:
            failures.append(role)
    if failures:
        _fail(f"bounded replay verification failed: {failures}")
    return 0


def stage_export(args: argparse.Namespace) -> int:
    from tdcsim_cbo._json import canonical_json_sha256, sha256_file
    from tdcsim_cbo.runtime_identity import distribution_identity, installed_archive_sha256
    from tdcsim_cbo.scenario_family_export import export_scenario_family

    _assert_installed_distribution_import()
    family, specs = _family(args)
    root = Path(args.output_root).resolve()
    runs = {}
    for role, scenario in specs:
        data = json.loads(scenario.read_text(encoding="utf-8"))
        verification_path = root / role / "verification.json"
        if not verification_path.is_file():
            _fail(f"{role}: verification record is missing")
        runs[role] = {
            "run_dir": root / role / "run",
            "scenario_id": data["scenario_id"],
            "scenario_sha256": canonical_json_sha256(data),
            "verification": json.loads(verification_path.read_text(encoding="utf-8")),
        }
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    if status:
        _fail("export requires a clean committed producer checkout")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    dist = distribution_identity()
    producer = {
        "repository_commit": commit,
        "installed_archive_sha256": installed_archive_sha256(),
        "distribution_file_digest": dist["file_digest"],
        "package_version": dist["version"],
        "python_version": platform.python_version(),
        "trc_routing_sha256": sha256_file(Path(args.trc_routing)),
    }
    receipt = export_scenario_family(
        family_spec=family,
        family_spec_sha256=sha256_file(Path(args.family_spec)),
        runs=runs,
        output_dir=Path(args.out_dir),
        producer=producer,
    )
    print(json.dumps({"output_dir": str(args.out_dir), "table_sha256": receipt["output"]["sha256"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, handler in (("run", stage_run), ("verify", stage_verify), ("export", stage_export)):
        stage = sub.add_parser(name)
        stage.add_argument("--family-spec", required=True)
        stage.add_argument("--scenario-dir", required=True)
        stage.add_argument("--output-root", required=True)
        if name in {"run", "export"}:
            stage.add_argument("--trc-routing", required=True)
        if name in {"run", "verify"}:
            stage.add_argument("--roles", help="comma-separated subset of family roles")
            stage.add_argument("--baseline", required=True, help="baseline package zip the scenarios were written for")
            stage.add_argument("--attestation", required=True)
        if name == "export":
            stage.add_argument("--out-dir", required=True)
        stage.set_defaults(handler=handler)
    one = sub.add_parser("verify-one")
    one.add_argument("--run-dir", required=True)
    one.add_argument("--out", required=True)
    one.add_argument("--baseline", required=True)
    one.add_argument("--attestation", required=True)
    one.set_defaults(handler=stage_verify_one)
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
