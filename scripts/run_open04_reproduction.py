#!/usr/bin/env python3
"""Reproduce the frozen OPEN-04 three-path campaign from committed source.

This is the external host-owned launcher for the OPEN-04 reproduction: the
retired in-package campaign controller is deliberately not reintroduced.  The
script drives the governed ``tdcsim_cbo`` APIs and CLI in four explicit
stages, each of which fails loudly rather than continuing past a defect:

``preflight``
    Regenerate the three canonical scenarios, gate them against the frozen
    boundary hashes, compile each role from the release-bound baseline
    package, re-freeze the campaign contract at the current producer commit,
    and run the pre-run verifier.
``run``
    Launch the three role runs in parallel (one bounded watchdog worker per
    role) with the full release-identity environment, importing the
    installed wheel distribution, then write the neutral host-task
    controller completion receipts.
``verify``
    Run the full post-run campaign verifier (bounded replay per role) and
    write the campaign verification receipt plus a replay-result cache for
    the export stage.
``export``
    Project the verified campaign into the four-file thin producer and
    consumer packages.

Every stage after ``preflight`` locates the campaign through the externally
supplied frozen contract SHA-256, never through a trusted local default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The frozen OPEN-04 economic boundary: canonical scenario hashes from the
# historical owner-signed contract.  Regenerated scenarios must reproduce
# these bytes exactly or the reproduction is not the same campaign.
FROZEN_SCENARIO_CANONICAL_SHA256 = {
    "baseline": (
        "59e2dd1395efbbe18e9d633c49140f642d0a487b0ca0c4e1381dc84dade5946d"
    ),
    "candidate_a": (
        "832424470210433e0344d3f5676c8b3be64acb8b7add5602a67abe309c677e86"
    ),
    "candidate_b": (
        "321fc0c76942edab370549d6b869d0a455c449c0807b8f9625aa0db0193a4746"
    ),
}
CAMPAIGN_ID_PREFIX = "open04-outcome-conditioned-three-path-v1"
ROLE_RUN_RELATIVE_PATHS = {
    "baseline": "baseline-role/baseline-run",
    "candidate_a": "candidate-a-role/candidate-a-run",
    "candidate_b": "candidate-b-role/candidate-b-run",
}
BASELINE_PACKAGE = PROJECT_ROOT / "output" / "cbo_forecast_release_bound_package.zip"
BASELINE_ATTESTATION = (
    PROJECT_ROOT / "output" / "cbo_forecast_release_bound_attestation.json"
)
PRE_RUN_RECEIPT_NAME = "open04_campaign_pre_run_receipt.json"
VERIFICATION_RECEIPT_NAME = "open04_campaign_verification_receipt.json"
CONTROLLER_SUMMARY_NAME = "open04_host_task_controller_summary.json"
REPLAY_CACHE_NAME = "open04_role_replay_results.json"
REPLAY_CACHE_SCHEMA = "tdcsim_open04_role_replay_cache_v1"
CONTROLLER_SUMMARY_SCHEMA = "tdcsim_open04_host_task_controller_summary_v1"
HOST_TASK_RECEIPT_SCHEMA = "tdcsim_open04_host_task_completion_receipt_v1"
RUN_MANIFEST_FILE = "tdcsim_cbo_run_manifest.json"
RUN_POLL_SECONDS = 10.0
TREE_TERMINATE_GRACE_SECONDS = 10.0
TREE_KILL_WAIT_SECONDS = 5.0
AGGREGATE_RSS_METHOD = "sum_of_role_peak_rss_upper_bound"


class ReproductionError(RuntimeError):
    """A stage precondition or postcondition failed; stop immediately."""


def _fail(message: str) -> None:
    raise ReproductionError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _assert_installed_distribution_import() -> None:
    """The producer and verifier must run the installed wheel, never src/."""

    import tdcsim_cbo

    package_file = Path(tdcsim_cbo.__file__).resolve()
    src_dir = (PROJECT_ROOT / "src").resolve()
    if src_dir in package_file.parents:
        _fail(
            "tdcsim_cbo was imported from the project source tree; run this "
            "script with the project virtual environment interpreter and "
            "without src/ on sys.path"
        )
    from tdcsim_cbo.runtime_identity import distribution_identity

    identity = distribution_identity()
    if identity["identity_source"] != "installed_distribution_files":
        _fail(
            "no installed tdcsim distribution was found; install the release "
            "wheel before running any stage"
        )


def _write_deterministic_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(payload)


def _short_sha(commit_sha: str) -> str:
    return commit_sha[:7]


def _runtime_dir_for(short: str) -> Path:
    return PROJECT_ROOT / "do" / f"open04_runtime_{short}"


def _open_baseline():
    from tdcsim_cbo.baseline import CboBaselinePackage

    for path in (BASELINE_PACKAGE, BASELINE_ATTESTATION):
        if not path.is_file():
            _fail(f"release-bound baseline dependency is missing: {path}")
    return CboBaselinePackage.open(
        BASELINE_PACKAGE, attestation_path=BASELINE_ATTESTATION
    )


def _build_code_identity(
    baseline: Any,
    wheel_path: Path,
) -> dict[str, Any]:
    """Compute the exact code identity the release-bound runner will record."""

    from tdcsim_cbo import runner as runner_module
    from tdcsim_cbo._json import sha256_file
    from tdcsim_cbo.consumer_challenge import collect_release_identity
    from tdcsim_cbo.runtime_identity import (
        distribution_identity,
        installed_archive_sha256,
        verify_wheel_against_git_commit,
        wheel_file_digest,
    )

    source = collect_release_identity(PROJECT_ROOT)
    commit = str(source["release_commit_sha"])
    if not wheel_path.is_file():
        _fail(f"release wheel is missing: {wheel_path}")
    wheel_sha = sha256_file(wheel_path)
    verify_wheel_against_git_commit(wheel_path, PROJECT_ROOT, commit)
    dist = distribution_identity()
    if dist["identity_source"] != "installed_distribution_files":
        _fail("code identity requires an installed distribution")
    if wheel_file_digest(wheel_path) != dist["file_digest"]:
        _fail(
            "release wheel runtime files differ from the installed "
            "distribution; reinstall the wheel"
        )
    if installed_archive_sha256() != wheel_sha:
        _fail(
            "installed distribution archive pin differs from the release "
            "wheel bytes; reinstall the wheel"
        )
    attested_lock = str(
        baseline.attestation.data.get("requirements_lock_sha256") or ""
    )
    locks = {
        record["relative_path"]: record
        for record in source["dependency_lock_files"]
    }
    if set(locks) != {"uv.lock", "requirements.lock.txt"}:
        _fail("producer commit must bind uv.lock and requirements.lock.txt")
    if locks["requirements.lock.txt"]["sha256"] != attested_lock:
        _fail(
            "committed requirements.lock.txt differs from the baseline "
            "attestation lock"
        )
    identity = {
        "code_commit_sha": commit,
        "dirty_state": False,
        "requirements_lock_sha256": attested_lock,
        "uv_lock_sha256": locks["uv.lock"]["sha256"],
        "dependency_lock_set_sha256": source["dependency_lock_set_sha256"],
        "wheel_sha256": wheel_sha,
        "wheel_artifact_sha256": wheel_sha,
        "runner_source_sha256": sha256_file(Path(runner_module.__file__)),
        "sim_engine_source_sha256": sha256_file(
            Path(runner_module.run_simulation.__code__.co_filename)
        ),
        **runner_module._open04_code_surface_hashes(),
        "python_version": platform.python_version(),
        "package_name": dist["name"],
        "package_version": dist["version"],
        "distribution_file_digest": dist["file_digest"],
        "runtime_identity_source": dist["identity_source"],
    }
    return identity


def _assert_frozen_runtime_identity(
    contract: Mapping[str, Any], runtime_dir: Path
) -> None:
    """Bind the current installed runtime to the frozen producer before work."""
    from tdcsim_cbo._json import sha256_file
    from tdcsim_cbo.runtime_identity import (
        distribution_identity,
        installed_archive_sha256,
        locked_environment_mismatches,
        verify_wheel_against_git_commit,
        wheel_file_digest,
    )

    common = contract["common_identity"]
    wheel = runtime_dir / f"tdcsim-{common['package_version']}-py3-none-any.whl"
    dist = distribution_identity()
    expected = {
        "name": common["package_name"],
        "version": common["package_version"],
        "file_digest": common["distribution_file_digest"],
        "identity_source": "installed_distribution_files",
    }
    if any(dist.get(key) != value for key, value in expected.items()):
        _fail("installed distribution differs from the frozen campaign identity")
    if not wheel.is_file() or sha256_file(wheel) != common["wheel_sha256"]:
        _fail("retained release wheel differs from the frozen campaign identity")
    if installed_archive_sha256() != common["wheel_sha256"]:
        _fail("installed archive differs from the frozen campaign identity")
    if wheel_file_digest(wheel) != common["distribution_file_digest"]:
        _fail("retained wheel runtime files differ from the frozen campaign identity")
    verify_wheel_against_git_commit(wheel, PROJECT_ROOT, common["code_commit_sha"])
    locks = []
    for name, key in (
        ("uv.lock", "uv_lock_sha256"),
        ("requirements.lock.txt", "requirements_lock_sha256"),
    ):
        payload = subprocess.run(
            ["git", "show", f"{common['code_commit_sha']}:{name}"],
            cwd=PROJECT_ROOT, check=True, capture_output=True,
        ).stdout
        digest = hashlib.sha256(payload).hexdigest()
        if digest != common[key]:
            _fail(f"frozen {name} digest differs from its producer commit")
        locks.append({"relative_path": name, "sha256": digest, "bytes": len(payload)})
        if name == "requirements.lock.txt":
            mismatches = locked_environment_mismatches(payload)
            if mismatches:
                _fail(f"installed dependencies differ from the frozen lock: {mismatches}")
    # Preserve the uv.lock, requirements.lock.txt order frozen by preflight.
    lock_digest = hashlib.sha256(json.dumps(
        locks, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if lock_digest != common["dependency_lock_set_sha256"]:
        _fail("dependency lock set differs from the frozen campaign identity")
    if platform.python_version() != common["python_version"]:
        _fail("Python version differs from the frozen campaign identity")


class _WindowsJob:
    """Contain descendants before a gated bootstrap can start the role."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in (
                "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
            )]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits), ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        self.ctypes = ctypes
        self.member_handles: dict[int, Any] = {}
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        for name, args, result in (
            ("CreateJobObjectW", [ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            ("SetInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            ("AssignProcessToJobObject", [wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            ("QueryInformationJobObject", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            ("TerminateJobObject", [wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            ("CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            ("OpenProcess", [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            ("WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        ):
            func = getattr(self.api, name)
            func.argtypes, func.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(
            self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)
        ):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def remember_members(self) -> bool:
        # Retain wait handles before termination: ActiveProcesses can become
        # zero before the member processes themselves become signalled.
        ctypes = self.ctypes
        capacity = 32
        while True:
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", ctypes.c_uint32), ("listed", ctypes.c_uint32),
                            ("ids", ctypes.c_size_t * capacity)]
            members = ProcessIds()
            if self.api.QueryInformationJobObject(
                self.handle, 3, ctypes.byref(members), ctypes.sizeof(members), None
            ):
                break
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, members.assigned)
        for pid in members.ids[:members.listed]:
            if pid in self.member_handles:
                continue
            handle = self.api.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if not handle:
                error = ctypes.get_last_error()
                if error == 87:  # The process exited before OpenProcess.
                    continue
                raise ctypes.WinError(error)
            self.member_handles[pid] = handle
        return members.assigned != 0

    def alive(self) -> bool:
        if self.handle is None:
            return False
        active = self.remember_members()
        for handle in self.member_handles.values():
            status = self.api.WaitForSingleObject(handle, 0)
            if status == 258:  # WAIT_TIMEOUT: termination is not complete.
                active = True
            elif status != 0:
                raise self.ctypes.WinError(self.ctypes.get_last_error())
        return active

    def kill(self) -> None:
        self.remember_members()
        if not self.api.TerminateJobObject(self.handle, 1):
            raise self.ctypes.WinError(self.ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.api.CloseHandle(self.handle)
            self.handle = None
        for handle in self.member_handles.values():
            self.api.CloseHandle(handle)
        self.member_handles.clear()


class _RoleProcess:
    """A role's OS containment boundary, including children after leader exit."""

    def __init__(self, command: list[str], **kwargs: Any) -> None:
        self.job = None
        self.process = None
        try:
            if os.name == "nt":
                from tdcsim_cbo.cli import _watchdog_worker_launch

                self.job = _WindowsJob()
                executable, venv_launcher = _watchdog_worker_launch()
                if venv_launcher is not None:
                    kwargs["env"] = {**kwargs.get("env", os.environ),
                                     "__PYVENV_LAUNCHER__": venv_launcher}
                # No role code or descendants execute before Job assignment.
                bootstrap = (
                    "import subprocess,sys; "
                    "token=sys.stdin.buffer.read(1); "
                    "sys.exit(subprocess.call(sys.argv[1:]) if token == b'1' else 1)"
                )
                self.process = subprocess.Popen(
                    [executable, "-B", "-c", bootstrap, *command],
                    stdin=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, **kwargs,
                )
                self.job.assign(self.process)
                self.process.stdin.write(b"1")
                self.process.stdin.close()
            else:
                self.process = subprocess.Popen(command, start_new_session=True, **kwargs)
        except BaseException:
            if self.process is not None:
                # A failed assignment leaves only the waiting bootstrap alive.
                if self.job is not None:
                    self.job.kill()
                self.process.kill()
                self.process.wait(timeout=5)
            if self.job is not None:
                self.job.close()
            raise

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def alive(self) -> bool:
        self.process.poll()  # reap our direct child before probing descendants
        if self.job is not None:
            return self.job.alive()
        # Exclude zombies: they cannot execute or write, and only their parent
        # (possibly the host's init) can reap them after group termination.
        result = subprocess.run(
            ["ps", "-axo", "pgid=,stat="], check=True, capture_output=True, text=True
        )
        return any(
            len(fields := line.split()) >= 2
            and fields[0] == str(self.pid) and not fields[1].startswith("Z")
            for line in result.stdout.splitlines()
        )

    def terminate(self) -> None:
        if self.job is not None:
            self.job.remember_members()
            try:
                self.process.send_signal(signal.CTRL_BREAK_EVENT)
            except OSError:
                pass  # Escalate the Job after the same bounded grace period.
        else:
            try:
                os.killpg(self.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def kill(self) -> None:
        if self.job is not None:
            self.job.kill()
        else:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        self.process.wait(timeout=5)
        if self.job is not None:
            self.job.close()


def _drain_role_processes(processes: Mapping[str, _RoleProcess]) -> None:
    """Bounded terminate, escalate, and positively verify every role tree."""
    for process in processes.values():
        process.terminate()
    for duration, escalate in ((TREE_TERMINATE_GRACE_SECONDS, True), (TREE_KILL_WAIT_SECONDS, False)):
        deadline = time.monotonic() + duration
        while any(process.alive() for process in processes.values()):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        survivors = [process for process in processes.values() if process.alive()]
        if not survivors:
            for process in processes.values():
                process.close()
            return
        if escalate:
            for process in survivors:
                process.kill()
    _fail("role process trees did not drain after terminate and kill deadlines")


def _aggregate_peak_rss_mb(role_evidence: Mapping[str, Mapping[str, Any]], budget: int) -> float:
    """Conservative aggregate upper bound, not a simultaneous host sample."""
    peak_bytes = sum(int(evidence["peak_rss_bytes"]) for evidence in role_evidence.values())
    if peak_bytes > budget:
        _fail(f"aggregate role peak RSS {peak_bytes} bytes exceeds acceptance budget {budget}")
    return peak_bytes / float(1024**2)


def _find_campaign(contract_sha256: str) -> tuple[Path, Path, dict[str, Any]]:
    """Locate the campaign whose frozen contract has the given SHA-256."""

    from tdcsim_cbo._json import sha256_file
    from tdcsim_cbo.open04_campaign import validate_open04_campaign_contract

    matches: list[Path] = []
    for candidate in sorted(
        (PROJECT_ROOT / "do").glob(
            "open04_runtime_*/campaign/open04_campaign_contract.json"
        )
    ):
        if sha256_file(candidate) == contract_sha256:
            matches.append(candidate)
    if len(matches) != 1:
        _fail(
            "expected exactly one frozen campaign contract matching "
            f"{contract_sha256}; found {len(matches)}"
        )
    contract_path = matches[0]
    campaign_root = contract_path.parent
    contract = validate_open04_campaign_contract(
        campaign_root, expected_contract_sha256=contract_sha256
    )
    return campaign_root, campaign_root.parent, contract


def _ensure_release_worktree(commit: str) -> Path:
    """Create or verify a clean detached worktree pinned at the commit."""

    worktree = PROJECT_ROOT / ".worktrees" / f"open04_release_{_short_sha(commit)}"
    if not worktree.exists():
        subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "--detach",
                str(worktree),
                commit,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != commit:
        _fail(
            f"release worktree {worktree} is at {head}, not the contract "
            f"commit {commit}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=worktree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        _fail(f"release worktree {worktree} is not clean:\n{status}")
    return worktree


def _release_environment(
    contract: Mapping[str, Any],
    campaign_root: Path,
    runtime_dir: Path,
    worktree: Path,
) -> dict[str, str]:
    import os

    from tdcsim_cbo._json import sha256_file
    from tdcsim_cbo.process_watchdog import THREAD_LIMIT_ENVIRONMENT_VARIABLES

    common = contract["common_identity"]
    wheel_path = runtime_dir / f"tdcsim-{common['package_version']}-py3-none-any.whl"
    if not wheel_path.is_file():
        _fail(f"retained release wheel is missing: {wheel_path}")
    if sha256_file(wheel_path) != common["wheel_sha256"]:
        _fail("retained release wheel bytes differ from the frozen contract")
    environment = dict(os.environ)
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        environment[name] = "1"
    environment.update(
        {
            "TDCSIM_CBO_CODE_COMMIT_SHA": common["code_commit_sha"],
            "TDCSIM_CBO_DIRTY_STATE": "false",
            "TDCSIM_CBO_SOURCE_REPOSITORY": str(worktree),
            "TDCSIM_CBO_WHEEL_PATH": str(wheel_path),
            "TDCSIM_CBO_WHEEL_SHA256": common["wheel_sha256"],
            "TDCSIM_CBO_CAMPAIGN_ROOT": str(campaign_root),
            "TDCSIM_CBO_CAMPAIGN_ID": contract["campaign_id"],
        }
    )
    return environment


def _apply_environment(environment: Mapping[str, str]) -> None:
    import os

    for name, value in environment.items():
        os.environ[name] = value


def _scan_stage_conflicts(campaign_root: Path, contract: Mapping[str, Any]) -> list[str]:
    """Count leftover claim/handoff/staging control files before launching."""

    conflicts: list[str] = []
    patterns = (".claim", "watchdog", ".staging-")
    for role in contract["execution_order"]:
        role_parent = (
            campaign_root / contract["roles"][role]["run_relative_path"]
        ).parent
        if not role_parent.is_dir():
            continue
        for entry in sorted(role_parent.iterdir()):
            name = entry.name
            if any(token in name for token in patterns):
                conflicts.append(str(entry))
    return conflicts


def stage_preflight(args: argparse.Namespace) -> int:
    from tdcsim_cbo._json import canonical_json_sha256, read_json, sha256_file
    from tdcsim_cbo.compiler import CboScenarioCompiler
    from tdcsim_cbo.contract import CboScenarioSpec
    from tdcsim_cbo.open04_campaign import (
        OPEN04_BASELINE_IDENTITY,
        _canonical_candidate_issuance_mix,
        build_open04_scenario_mappings,
        freeze_open04_campaign_contract,
        verify_open04_campaign_pre_run,
    )
    from tdcsim_cbo.consumer_challenge import collect_release_identity
    from tdcsim_cbo.runtime_identity import locked_environment_mismatches

    _assert_installed_distribution_import()
    source = collect_release_identity(PROJECT_ROOT)
    commit = str(source["release_commit_sha"])
    short = _short_sha(commit)
    runtime_dir = _runtime_dir_for(short)
    campaign_root = runtime_dir / "campaign"
    campaign_id = f"{CAMPAIGN_ID_PREFIX}-{short}"
    if campaign_root.exists():
        _fail(
            f"campaign root already exists: {campaign_root}; remove the "
            "runtime directory to redo preflight"
        )

    lock_mismatches = locked_environment_mismatches(
        (PROJECT_ROOT / "requirements.lock.txt").read_bytes()
    )
    if lock_mismatches:
        _fail(
            "installed environment does not match requirements.lock.txt: "
            f"{lock_mismatches}"
        )

    baseline = _open_baseline()
    declared = {
        "package_id": baseline.package_id,
        "package_sha256": baseline.package_sha256,
        "manifest_sha256": baseline.manifest_sha256,
        "release_attestation_sha256": baseline.attestation.sha256,
    }
    if declared != dict(OPEN04_BASELINE_IDENTITY):
        _fail(
            "local baseline package/attestation identity differs from the "
            f"authoritative OPEN-04 baseline: {declared}"
        )

    source_wheel = Path(args.wheel).expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    retained_wheel = runtime_dir / source_wheel.name
    if retained_wheel.exists():
        if sha256_file(retained_wheel) != sha256_file(source_wheel):
            _fail(
                f"retained wheel {retained_wheel} differs from {source_wheel}"
            )
    else:
        retained_wheel.write_bytes(source_wheel.read_bytes())
    code_identity = _build_code_identity(baseline, retained_wheel)
    if code_identity["code_commit_sha"] != commit:
        _fail("code identity commit differs from the repository HEAD")

    scenarios = build_open04_scenario_mappings(
        baseline_identity=OPEN04_BASELINE_IDENTITY,
        candidate_a_issuance_mix=_canonical_candidate_issuance_mix(
            "candidate_a"
        ),
        candidate_b_issuance_mix=_canonical_candidate_issuance_mix(
            "candidate_b"
        ),
    )
    for role, scenario in scenarios.items():
        actual = canonical_json_sha256(scenario)
        expected = FROZEN_SCENARIO_CANONICAL_SHA256[role]
        if actual != expected:
            _fail(
                f"{role} regenerated scenario hash {actual} differs from the "
                f"frozen boundary {expected}"
            )
    print("boundary gate: all three canonical scenario hashes match")

    campaign_root.mkdir(parents=True)
    compiler = CboScenarioCompiler()
    compiled_dirs: dict[str, Path] = {}
    for role, scenario in scenarios.items():
        compiled = compiler.compile(
            baseline,
            CboScenarioSpec.from_mapping(scenario),
            campaign_root / f"pre-run-compile-{role}",
        )
        compiled_dirs[role] = compiled.compiled_dir
        print(
            f"compiled {role}: inputs digest {compiled.compiled_inputs_digest}"
        )
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

    signature_reference = str(args.signature_reference).strip()
    if not signature_reference:
        _fail("preflight requires a nonempty --signature-reference")
    contract = freeze_open04_campaign_contract(
        campaign_root,
        campaign_id=campaign_id,
        signature_reference=signature_reference,
        scenarios_by_role=scenarios,
        compiled_dirs_by_role=compiled_dirs,
        code_identity=code_identity,
        role_to_run_relative_path=ROLE_RUN_RELATIVE_PATHS,
        canonical_issuance_mix_sha256_by_role=mix_hashes,
    )
    contract_path = campaign_root / "open04_campaign_contract.json"
    contract_sha = sha256_file(contract_path)

    receipt = verify_open04_campaign_pre_run(
        campaign_root,
        expected_contract_sha256=contract_sha,
        compiled_dirs_by_role=compiled_dirs,
    )
    if receipt["status"] != "pass":
        _fail(f"pre-run verification did not pass: {receipt}")
    _write_deterministic_json(campaign_root / PRE_RUN_RECEIPT_NAME, receipt)

    print("preflight: pass")
    print(f"signature_reference: {contract['signature_reference']}")
    print(f"campaign_id: {contract['campaign_id']}")
    print(f"campaign_root: {campaign_root}")
    print(f"contract_sha256: {contract_sha}")
    print(f"code_commit_sha: {commit}")
    print(f"wheel_sha256: {code_identity['wheel_sha256']}")
    print(
        "next: "
        f"{shlex.quote(sys.executable)} scripts/run_open04_reproduction.py "
        f"run --contract-sha256 {contract_sha}"
    )
    return 0


def stage_run(args: argparse.Namespace) -> int:
    from tdcsim_cbo._json import read_json, sha256_file

    _assert_installed_distribution_import()
    campaign_root, runtime_dir, contract = _find_campaign(args.contract_sha256)
    _assert_frozen_runtime_identity(contract, runtime_dir)
    common = contract["common_identity"]
    worktree = _ensure_release_worktree(common["code_commit_sha"])
    environment = _release_environment(
        contract, campaign_root, runtime_dir, worktree
    )

    roles = list(contract["execution_order"])
    for role in roles:
        run_root = campaign_root / contract["roles"][role]["run_relative_path"]
        if run_root.exists():
            _fail(
                f"{role} run output already exists: {run_root}; remove the "
                "role parents from any failed attempt before relaunching"
            )
        receipt_path = campaign_root / contract["roles"][role][
            "controller_completion_receipt_relative_path"
        ]
        if receipt_path.exists():
            _fail(f"{role} completion receipt already exists: {receipt_path}")
    conflicts = _scan_stage_conflicts(campaign_root, contract)
    preflight_conflicts = len(conflicts)
    if preflight_conflicts:
        _fail(
            "stale run-control files present; remove them before "
            f"launching: {conflicts}"
        )

    log_dir = runtime_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    batch_started = _utc_now()
    controller_run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{platform.node()}-open04-{contract['campaign_id']}"
    )
    processes: dict[str, _RoleProcess] = {}
    commands: dict[str, list[str]] = {}
    started_at: dict[str, str] = {}
    completed_at: dict[str, str] = {}
    logs: dict[str, Any] = {}
    failures: dict[str, int] = {}
    pending = set(roles)
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"launcher interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for role in roles:
            run_root = campaign_root / contract["roles"][role]["run_relative_path"]
            scenario_path = campaign_root / contract["roles"][role][
                "scenario_source_relative_path"
            ]
            command = [
                sys.executable,
                "-B",
                "-m",
                "tdcsim_cbo.cli",
                "run",
                "--baseline",
                str(BASELINE_PACKAGE),
                "--attestation",
                str(BASELINE_ATTESTATION),
                "--scenario",
                str(scenario_path),
                "--output-dir",
                str(run_root),
            ]
            commands[role] = command
            log_path = log_dir / f"{role}.log"
            log_handle = log_path.open("ab")
            logs[role] = log_handle
            started_at[role] = _utc_now()
            processes[role] = _RoleProcess(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            print(f"launched {role}: pid {processes[role].pid} -> {log_path}")

        while pending:
            for role in sorted(pending):
                code = processes[role].poll()
                if code is None:
                    continue
                pending.discard(role)
                completed_at[role] = _utc_now()
                if code == 0:
                    print(f"{role}: completed with exit code 0")
                else:
                    failures[role] = code
                    print(f"{role}: FAILED with exit code {code}")
            if failures:
                break
            if pending:
                time.sleep(RUN_POLL_SECONDS)
    finally:
        # A second interrupt must not interrupt the bounded drain itself.
        previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            _drain_role_processes(processes)
        finally:
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGTERM, previous_term)
            for handle in logs.values():
                handle.close()
    if failures:
        _fail(
            f"role runs failed: {failures}; see logs under {log_dir}; no "
            "completion receipts were written"
        )

    exit_confirmed: dict[str, str] = {}
    role_evidence: dict[str, dict[str, Any]] = {}
    for role in roles:
        run_root = campaign_root / contract["roles"][role]["run_relative_path"]
        leftovers = [
            str(entry)
            for entry in sorted(run_root.parent.iterdir())
            if entry.name.startswith(f".{run_root.name}.")
        ]
        if leftovers:
            _fail(
                f"{role} worker left run-control residue after a successful "
                f"exit: {leftovers}"
            )
        manifest_path = run_root / RUN_MANIFEST_FILE
        if not manifest_path.is_file():
            _fail(f"{role} run manifest is missing after a successful exit")
        manifest = read_json(manifest_path)
        if manifest.get("status") != "complete":
            _fail(f"{role} run manifest status is not complete")
        summary_record = manifest["output_manifest"]["summary"]
        summary_path = run_root / "outputs" / "summary.json"
        summary_sha = sha256_file(summary_path)
        if summary_sha != summary_record["sha256"]:
            _fail(f"{role} terminal summary differs from its run manifest")
        exit_confirmed[role] = _utc_now()
        role_evidence[role] = {
            "run_id": str(manifest["run_id"]),
            "run_manifest_sha256": sha256_file(manifest_path),
            "terminal_summary_sha256": summary_sha,
            "peak_rss_bytes": int(
                manifest["bounded_evidence"]["peak_rss_bytes"]
            ),
        }

    peak_rss_mb = _aggregate_peak_rss_mb(
        role_evidence,
        int(contract["execution_contract"]["aggregate_acceptance_peak_rss_bytes"]),
    )

    summary = {
        "schema_version": CONTROLLER_SUMMARY_SCHEMA,
        "campaign_id": contract["campaign_id"],
        "campaign_contract_sha256": args.contract_sha256,
        "controller": "host_owned_task",
        "controller_run_id": controller_run_id,
        "host": platform.node(),
        "batch_started_at_utc": batch_started,
        "batch_completed_at_utc": _utc_now(),
        "controller_peak_rss_mb": peak_rss_mb,
        "controller_peak_rss_method": AGGREGATE_RSS_METHOD,
        "roles": {
            role: {
                "command": commands[role],
                "started_at_utc": started_at[role],
                "completed_at_utc": completed_at[role],
                "worker_exit_confirmed_at_utc": exit_confirmed[role],
                "exit_code": 0,
                **role_evidence[role],
            }
            for role in roles
        },
    }
    summary_path = runtime_dir / CONTROLLER_SUMMARY_NAME
    _write_deterministic_json(summary_path, summary)
    controller_summary_sha256 = sha256_file(summary_path)

    for role in roles:
        command_text = " ".join(shlex.quote(part) for part in commands[role])
        receipt = {
            "schema_version": HOST_TASK_RECEIPT_SCHEMA,
            "campaign_id": contract["campaign_id"],
            "campaign_contract_sha256": args.contract_sha256,
            "role": role,
            "run_id": role_evidence[role]["run_id"],
            "host": platform.node(),
            "controller": "host_owned_task",
            "placement": "auto_selected_host",
            "terminal_status": "completed",
            "started_at_utc": started_at[role],
            "completed_at_utc": completed_at[role],
            "worker_exit_confirmed_at_utc": exit_confirmed[role],
            "terminal_summary_sha256": role_evidence[role][
                "terminal_summary_sha256"
            ],
            "run_manifest_sha256": role_evidence[role][
                "run_manifest_sha256"
            ],
            "controller_run_id": controller_run_id,
            "controller_summary_sha256": controller_summary_sha256,
            "controller_command_sha256": hashlib.sha256(
                command_text.encode("utf-8")
            ).hexdigest(),
            "controller_command_exit_code": 0,
            "controller_exit_code": 0,
            "preflight_conflicts": preflight_conflicts,
            "postrun_conflicts": 0,
            "controller_peak_rss_mb": peak_rss_mb,
            "controller_avg_cpu_pct": None,
            "controller_memory_guard_status": (
                "accepted_bounded_role_watchdogs"
            ),
            "controller_telemetry_status": "bounded_role_watchdogs",
            "controller_telemetry_locator": (
                f"controller_summary_sha256:{controller_summary_sha256}"
            ),
            "controller_process_tree_drained": True,
        }
        receipt_path = campaign_root / contract["roles"][role][
            "controller_completion_receipt_relative_path"
        ]
        _write_deterministic_json(receipt_path, receipt)
        print(f"wrote {role} completion receipt: {receipt_path}")

    print("run: all three roles completed")
    print(
        "next: "
        f"{shlex.quote(sys.executable)} scripts/run_open04_reproduction.py "
        f"verify --contract-sha256 {args.contract_sha256}"
    )
    return 0


def stage_verify(args: argparse.Namespace) -> int:
    _assert_installed_distribution_import()
    campaign_root, runtime_dir, contract = _find_campaign(args.contract_sha256)
    _assert_frozen_runtime_identity(contract, runtime_dir)

    import tdcsim_cbo.verifier as verifier_module
    from tdcsim_cbo.open04_campaign import verify_open04_campaign_post_run
    common = contract["common_identity"]
    worktree = _ensure_release_worktree(common["code_commit_sha"])
    _apply_environment(
        _release_environment(contract, campaign_root, runtime_dir, worktree)
    )

    recorded: dict[str, dict[str, Any]] = {}
    original_verify = verifier_module.verify_scenario_run

    def recording_verify(run_dir, **kwargs):
        result = original_verify(run_dir, **kwargs)
        recorded[str(Path(run_dir).resolve())] = result
        return result

    verifier_module.verify_scenario_run = recording_verify
    try:
        receipt = verify_open04_campaign_post_run(
            campaign_root,
            expected_contract_sha256=args.contract_sha256,
            baseline_package=BASELINE_PACKAGE,
            attestation=BASELINE_ATTESTATION,
        )
    finally:
        verifier_module.verify_scenario_run = original_verify
    if receipt["status"] != "pass" or receipt["campaign_eligible"] is not True:
        _fail(f"post-run verification did not pass: {receipt}")

    receipt_path = campaign_root / VERIFICATION_RECEIPT_NAME
    _write_deterministic_json(receipt_path, receipt)
    print(f"wrote campaign verification receipt: {receipt_path}")

    from tdcsim_cbo._json import sha256_file

    cache_roles: dict[str, Any] = {}
    for role in contract["execution_order"]:
        run_root = (
            campaign_root / contract["roles"][role]["run_relative_path"]
        ).resolve()
        result = recorded.get(str(run_root))
        if result is None:
            _fail(
                f"{role} bounded replay result was not captured; the replay "
                "cache cannot be written"
            )
        cache_roles[role] = {
            "run_manifest_sha256": sha256_file(run_root / RUN_MANIFEST_FILE),
            "result": result,
        }
    cache = {
        "schema_version": REPLAY_CACHE_SCHEMA,
        "campaign_contract_sha256": args.contract_sha256,
        "roles": cache_roles,
    }
    _write_deterministic_json(runtime_dir / REPLAY_CACHE_NAME, cache)

    print("verify: pass (campaign eligible)")
    print(
        "next: "
        f"{shlex.quote(sys.executable)} scripts/run_open04_reproduction.py "
        f"export --contract-sha256 {args.contract_sha256} "
        "--consumer-output-dir <consumer destination>"
    )
    return 0


def stage_export(args: argparse.Namespace) -> int:
    from tdcsim_cbo._json import read_json, sha256_file
    _assert_installed_distribution_import()
    campaign_root, runtime_dir, contract = _find_campaign(args.contract_sha256)
    _assert_frozen_runtime_identity(contract, runtime_dir)

    from tdcsim_cbo.open04_export import export_open04_thin_package
    common = contract["common_identity"]
    worktree = _ensure_release_worktree(common["code_commit_sha"])
    _apply_environment(
        _release_environment(contract, campaign_root, runtime_dir, worktree)
    )

    contract_path = campaign_root / "open04_campaign_contract.json"
    receipt_path = campaign_root / VERIFICATION_RECEIPT_NAME
    if not receipt_path.is_file():
        _fail(
            f"campaign verification receipt is missing: {receipt_path}; run "
            "the verify stage first"
        )
    run_roots = {
        role: campaign_root / contract["roles"][role]["run_relative_path"]
        for role in contract["execution_order"]
    }

    verified_run_results: dict[str, Mapping[str, Any]] | None = None
    cache_path = runtime_dir / REPLAY_CACHE_NAME
    if cache_path.is_file():
        cache = read_json(cache_path)
        if (
            cache.get("schema_version") != REPLAY_CACHE_SCHEMA
            or cache.get("campaign_contract_sha256") != args.contract_sha256
        ):
            _fail(f"replay cache does not bind this campaign: {cache_path}")
        verified_run_results = {}
        for role, root in run_roots.items():
            entry = cache["roles"][role]
            if (
                entry["run_manifest_sha256"]
                != sha256_file(root / RUN_MANIFEST_FILE)
            ):
                _fail(
                    f"{role} run manifest changed after verification; rerun "
                    "the verify stage"
                )
            verified_run_results[role] = entry["result"]
    else:
        print(
            "WARNING: no replay cache found; the exporter will re-run the "
            "bounded replay verification for each role"
        )

    short = _short_sha(common["code_commit_sha"])
    producer_out = (
        Path(args.producer_output_dir).expanduser().resolve()
        if args.producer_output_dir
        else PROJECT_ROOT / "output" / "evidence" / "open04_acceptance" / short
    )
    consumer_out = Path(args.consumer_output_dir).expanduser().resolve()
    producer_out.parent.mkdir(parents=True, exist_ok=True)
    consumer_out.parent.mkdir(parents=True, exist_ok=True)

    result = export_open04_thin_package(
        run_roots,
        contract_path,
        receipt_path,
        producer_out,
        consumer_out,
        consumer_project=args.consumer_project,
        verified_run_results=verified_run_results,
    )
    if result.receipt.get("overall_status") != "pass":
        _fail(f"thin export did not pass: {result.receipt}")
    print("export: pass")
    print(f"producer_output_dir: {result.producer_output_dir}")
    print(f"consumer_output_dir: {result.consumer_output_dir}")
    for artifact in (
        result.tdc_paths,
        result.maturity_metrics,
        result.scenario_input_contract,
        result.producer_consumer_receipt,
    ):
        print(f"producer artifact: {artifact} ({sha256_file(artifact)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run_open04_reproduction",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="stage", required=True)

    preflight = sub.add_parser(
        "preflight",
        help="freeze and pre-run-verify the campaign at the current commit",
    )
    preflight.add_argument(
        "--wheel",
        default=str(PROJECT_ROOT / "dist" / "tdcsim-0.1.0-py3-none-any.whl"),
        help="built release wheel to retain and bind (default: dist wheel)",
    )
    preflight.add_argument(
        "--signature-reference",
        required=True,
        help=(
            "owner signature reference for the frozen contract, mirroring "
            "the historical contract's double-underscore reference format "
            "and recording the current owner authorization"
        ),
    )
    preflight.set_defaults(handler=stage_preflight)

    run = sub.add_parser(
        "run", help="launch the three role runs and write completion receipts"
    )
    run.add_argument("--contract-sha256", required=True)
    run.set_defaults(handler=stage_run)

    verify = sub.add_parser(
        "verify", help="run the post-run campaign verifier (bounded replay)"
    )
    verify.add_argument("--contract-sha256", required=True)
    verify.set_defaults(handler=stage_verify)

    export = sub.add_parser(
        "export", help="export the four-file thin producer/consumer packages"
    )
    export.add_argument("--contract-sha256", required=True)
    export.add_argument(
        "--consumer-output-dir",
        required=True,
        help="absent consumer destination directory for the thin package",
    )
    export.add_argument(
        "--producer-output-dir",
        default=None,
        help=(
            "absent producer destination directory (default: "
            "output/evidence/open04_acceptance/<short-sha>)"
        ),
    )
    export.add_argument("--consumer-project", default="thesis")
    export.set_defaults(handler=stage_export)

    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
