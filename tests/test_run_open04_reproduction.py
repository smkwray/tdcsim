from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tdcsim_cbo.runtime_identity import distribution_identity as measured_distribution_identity


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/run_open04_reproduction.py"
    spec = importlib.util.spec_from_file_location("open04_launcher_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parallel_memory_gate_uses_aggregate_not_max(launcher):
    gib = 1024**3
    with pytest.raises(launcher.ReproductionError, match="aggregate"):
        launcher._aggregate_peak_rss_mb({str(i): {"peak_rss_bytes": 4 * gib} for i in range(3)}, 10 * gib)
    peaks = [651_362_304, 895_746_048, 942_899_200]
    assert launcher._aggregate_peak_rss_mb(
        {str(i): {"peak_rss_bytes": value} for i, value in enumerate(peaks)}, 10 * gib
    ) == 2374.65625
    assert launcher.AGGREGATE_RSS_METHOD == "sum_of_role_peak_rss_upper_bound"


def _wait_for(path):
    deadline = time.monotonic() + 10
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"child did not start: {path}")
        time.sleep(0.02)


def _alive(pid):
    if os.name == "nt":
        import ctypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        api.OpenProcess.restype = ctypes.c_void_p
        api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        api.WaitForSingleObject.restype = ctypes.c_ulong
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def _prepare_run(launcher, monkeypatch, tmp_path, failure):
    roles = ["baseline", "candidate_a", "candidate_b"]
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    contract = {
        "common_identity": {"code_commit_sha": "a" * 40}, "execution_order": roles,
        "campaign_id": "synthetic-tree-drain",
        "roles": {role: {
            "run_relative_path": f"{role}/run",
            "controller_completion_receipt_relative_path": f"{role}.json",
            "scenario_source_relative_path": f"{role}.yaml",
        } for role in roles},
    }
    monkeypatch.setattr(launcher, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(launcher, "_assert_installed_distribution_import", lambda: None)
    monkeypatch.setattr(launcher, "_assert_frozen_runtime_identity", lambda *args: None)
    monkeypatch.setattr(launcher, "_find_campaign", lambda _: (campaign, tmp_path, contract))
    monkeypatch.setattr(launcher, "_ensure_release_worktree", lambda _: tmp_path)
    monkeypatch.setattr(launcher, "_release_environment", lambda *args: dict(os.environ))
    monkeypatch.setattr(launcher, "RUN_POLL_SECONDS", 0.03)
    monkeypatch.setattr(launcher, "TREE_TERMINATE_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(launcher, "TREE_KILL_WAIT_SECONDS", 5)
    real_process = launcher._RoleProcess
    created = []
    child_pids = []
    child = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "signal.signal(getattr(signal,'SIGBREAK',signal.SIGINT), signal.SIG_IGN); "
        "Path(__import__('sys').argv[1]).write_text(str(os.getpid())); time.sleep(60)"
    )

    def launch(command, **kwargs):
        index = len(created)
        if failure == "partial" and index == 1:
            raise OSError("synthetic launch failure")
        pid_path = tmp_path / f"child-{index}.pid"
        parent_path = tmp_path / f"outer-{index}.pid"
        outer = (
            "import os,signal,subprocess,sys,time; from pathlib import Path; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "signal.signal(getattr(signal,'SIGBREAK',signal.SIGINT), signal.SIG_IGN); "
            f"Path({str(parent_path)!r}).write_text(str(os.getpid())); "
            f"subprocess.Popen([sys.executable,'-c',{child!r},{str(pid_path)!r}]); "
            + ("time.sleep(0.3); sys.exit(7)" if failure == "sibling" and index == 1
               else "time.sleep(0.3); sys.exit(0)" if failure == "success" else "time.sleep(60)")
        )
        process = real_process([sys.executable, "-B", "-c", outer], **kwargs)
        created.append(process)
        _wait_for(pid_path)
        child_pids.append(int(pid_path.read_text()))
        child_pids.append(int(parent_path.read_text()))
        return process

    monkeypatch.setattr(launcher, "_RoleProcess", launch)
    if failure in {"interrupt", "exception"}:
        original_sleep = launcher.time.sleep
        raised = False

        def interrupted_sleep(seconds):
            nonlocal raised
            if seconds == launcher.RUN_POLL_SECONDS and not raised:
                raised = True
                if failure == "interrupt":
                    raise KeyboardInterrupt("synthetic interrupt")
                raise RuntimeError("synthetic monitor exception")
            original_sleep(seconds)

        monkeypatch.setattr(launcher.time, "sleep", interrupted_sleep)
    return campaign, created, child_pids


def _assert_failure_drains_real_process_tree(launcher, monkeypatch, tmp_path, failure):
    campaign, processes, pids = _prepare_run(launcher, monkeypatch, tmp_path, failure)
    expected = {"sibling": launcher.ReproductionError, "partial": OSError,
                "interrupt": KeyboardInterrupt, "exception": RuntimeError}[failure]
    try:
        with pytest.raises(expected):
            launcher.stage_run(argparse.Namespace(contract_sha256="a" * 64))
        assert processes and pids
        assert all(process.poll() is not None for process in processes)
        assert all(not _alive(pid) for pid in pids)
        assert not list(campaign.glob("*.json"))
        assert not (tmp_path / launcher.CONTROLLER_SUMMARY_NAME).exists()
    finally:
        # Failure of the assertions must not leak the synthetic test workers.
        for process in processes:
            if process.alive():
                process.kill()
        for process in processes:
            process.close()


def test_sibling_failure_drains_process_tree(launcher, monkeypatch, tmp_path):
    _assert_failure_drains_real_process_tree(launcher, monkeypatch, tmp_path, "sibling")


def test_partial_launch_failure_drains_started_tree(launcher, monkeypatch, tmp_path):
    _assert_failure_drains_real_process_tree(launcher, monkeypatch, tmp_path, "partial")


def test_interrupt_drains_process_tree(launcher, monkeypatch, tmp_path):
    _assert_failure_drains_real_process_tree(launcher, monkeypatch, tmp_path, "interrupt")


def test_monitor_exception_drains_process_tree(launcher, monkeypatch, tmp_path):
    _assert_failure_drains_real_process_tree(launcher, monkeypatch, tmp_path, "exception")


@pytest.fixture
def frozen_identity(launcher, monkeypatch, tmp_path):
    import tdcsim_cbo.runtime_identity as identity

    wheel = tmp_path / "tdcsim-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"retained frozen wheel")
    lock = b"example==1.0\n"
    uv = b"frozen uv lock\n"
    def digest(payload):
        return hashlib.sha256(payload).hexdigest()
    records = [{"relative_path": name, "sha256": digest(payload), "bytes": len(payload)}
               for name, payload in [("uv.lock", uv), ("requirements.lock.txt", lock)]]
    common = {
        "package_name": "tdcsim", "package_version": "0.1.0",
        "distribution_file_digest": "d" * 64, "wheel_sha256": digest(wheel.read_bytes()),
        "code_commit_sha": "a" * 40, "requirements_lock_sha256": digest(lock),
        "uv_lock_sha256": digest(uv), "python_version": launcher.platform.python_version(),
        "dependency_lock_set_sha256": digest(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()),
    }
    dist = {"name": "tdcsim", "version": "0.1.0", "file_digest": "d" * 64,
            "identity_source": "installed_distribution_files"}
    monkeypatch.setattr(identity, "distribution_identity", lambda: dict(dist))
    monkeypatch.setattr(identity, "installed_archive_sha256", lambda: common["wheel_sha256"])
    monkeypatch.setattr(identity, "wheel_file_digest", lambda _: "d" * 64)
    monkeypatch.setattr(identity, "verify_wheel_against_git_commit", lambda *args: {})
    monkeypatch.setattr(identity, "locked_environment_mismatches", lambda _: {})
    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kw: argparse.Namespace(
        stdout=uv if command[-1].endswith(":uv.lock") else lock))
    return {"common_identity": common}, dist, identity


def test_frozen_runtime_identity_accepts_matching_runtime(launcher, frozen_identity, tmp_path):
    contract, _, _ = frozen_identity
    launcher._assert_frozen_runtime_identity(contract, tmp_path)


@pytest.mark.parametrize("field", ["name", "version", "file_digest", "identity_source"])
def test_frozen_runtime_rejects_distribution_identity_drift(launcher, frozen_identity, tmp_path, field):
    contract, dist, _ = frozen_identity
    dist[field] = "changed"
    with pytest.raises(launcher.ReproductionError, match="installed distribution"):
        launcher._assert_frozen_runtime_identity(contract, tmp_path)


@pytest.mark.parametrize("helper", ["installed_archive_sha256", "wheel_file_digest", "locked_environment_mismatches"])
def test_frozen_runtime_rejects_archive_or_dependency_drift(launcher, frozen_identity, monkeypatch, tmp_path, helper):
    contract, _, identity = frozen_identity
    monkeypatch.setattr(identity, helper, lambda *args: "changed")
    with pytest.raises(launcher.ReproductionError):
        launcher._assert_frozen_runtime_identity(contract, tmp_path)


@pytest.mark.parametrize("stage", ["run", "verify", "export"])
def test_export_rejects_installed_distribution_drift(launcher, frozen_identity, monkeypatch, tmp_path, stage):
    contract, dist, _ = frozen_identity
    helper_file = tmp_path / "installed_helper.py"
    helper_file.write_text("original helper")
    exporter_file = tmp_path / "open04_export.py"
    exporter_file.write_text("unchanged exporter")
    before_exporter = exporter_file.read_bytes()
    # Exercise the production file-digest implementation on installed-file
    # metadata, then mutate a helper while keeping the exporter byte-identical.
    import tdcsim_cbo.runtime_identity as identity
    installed = argparse.Namespace(
        metadata={"Name": "tdcsim"}, version="0.1.0",
        files=[Path("installed_helper.py"), Path("open04_export.py")],
        locate_file=lambda item: tmp_path / item,
    )
    monkeypatch.setattr(identity.metadata, "distribution", lambda _: installed)
    monkeypatch.setattr(identity, "distribution_identity", measured_distribution_identity)
    contract["common_identity"]["distribution_file_digest"] = measured_distribution_identity()["file_digest"]
    helper_file.write_text("altered helper")
    monkeypatch.setattr(launcher, "_assert_installed_distribution_import", lambda: None)
    monkeypatch.setattr(launcher, "_find_campaign", lambda _: (tmp_path / "campaign", tmp_path, contract))
    monkeypatch.setattr(launcher, "_ensure_release_worktree", lambda _: pytest.fail("created a worktree before identity gate"))
    args = argparse.Namespace(contract_sha256="a" * 64, producer_output_dir=str(tmp_path / "producer" / "result"),
                              consumer_output_dir=str(tmp_path / "consumer" / "result"), consumer_project="synthetic")
    with pytest.raises(launcher.ReproductionError, match="installed distribution"):
        getattr(launcher, f"stage_{stage}")(args)
    assert exporter_file.read_bytes() == before_exporter
    assert not (tmp_path / "producer").exists()
    assert not (tmp_path / "consumer").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("peaks,accepted", [
    ([651_362_304, 895_746_048, 942_899_200], True),
    ([4 * 1024**3] * 3, False),
])
def test_run_receipts_bind_aggregate_and_positive_tree_drain(launcher, monkeypatch, tmp_path, peaks, accepted):
    from tdcsim_cbo.open04_campaign import _controller_completion_receipt

    campaign, processes, pids = _prepare_run(launcher, monkeypatch, tmp_path, "success")
    _, _, contract = launcher._find_campaign("a" * 64)
    contract["execution_contract"] = {"aggregate_acceptance_peak_rss_bytes": 10 * 1024**3}
    start = launcher._RoleProcess
    evidence = {}

    def launch_with_synthetic_manifest(command, **kwargs):
        process = start(command, **kwargs)
        root = Path(command[command.index("--output-dir") + 1])
        (root / "outputs").mkdir(parents=True)
        summary = root / "outputs/summary.json"
        summary.write_text('{}\n')
        summary_sha = hashlib.sha256(summary.read_bytes()).hexdigest()
        manifest = root / launcher.RUN_MANIFEST_FILE
        manifest.write_text(json.dumps({
            "status": "complete", "run_id": root.parent.name,
            "output_manifest": {"summary": {"sha256": summary_sha}},
            "bounded_evidence": {"peak_rss_bytes": peaks[len(evidence)]},
        }))
        evidence[root.parent.name] = (summary_sha, hashlib.sha256(manifest.read_bytes()).hexdigest())
        return process

    monkeypatch.setattr(launcher, "_RoleProcess", launch_with_synthetic_manifest)
    if accepted:
        assert launcher.stage_run(argparse.Namespace(contract_sha256="a" * 64)) == 0
        summary = json.loads((tmp_path / launcher.CONTROLLER_SUMMARY_NAME).read_text())
        assert summary["controller_peak_rss_method"] == "sum_of_role_peak_rss_upper_bound"
        assert summary["controller_peak_rss_mb"] == sum(peaks) / 1024**2
        for role, (summary_sha, manifest_sha) in evidence.items():
            receipt = _controller_completion_receipt(
                campaign / f"{role}.json", role=role, campaign_id=contract["campaign_id"],
                campaign_contract_sha256="a" * 64, run_id=role,
                run_manifest_sha256=manifest_sha, expected_terminal_summary_sha256=summary_sha,
            )
            assert receipt["controller_peak_rss_mb"] == summary["controller_peak_rss_mb"]
            assert receipt["controller_process_tree_drained"] is True
    else:
        with pytest.raises(launcher.ReproductionError, match="aggregate"):
            launcher.stage_run(argparse.Namespace(contract_sha256="a" * 64))
        assert not list(campaign.glob("*.json"))
        assert not (tmp_path / launcher.CONTROLLER_SUMMARY_NAME).exists()
    assert all(process.poll() is not None for process in processes)
    assert all(not _alive(pid) for pid in pids)
