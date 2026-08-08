"""Focused tests for the bounded CBO runner's claim and promotion protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tdcsim_cbo.bounded_output as bounded_output
import tdcsim_cbo.runner as runner
import test_tdcsim_cbo_closeout_interface as cbo_fixtures
from tdcsim_cbo import CboScenarioSpec
from tdcsim_cbo._json import read_json
from tdcsim_cbo.bounded_output import BoundedResourceLimits
from tdcsim_cbo.process_watchdog import (
    GIB,
    THREAD_LIMIT_ENVIRONMENT_VARIABLES,
    WatchdogResult,
)
from tdcsim_cbo.runner import (
    RunnerError,
    cleanup_watchdog_intervention,
    consume_watchdog_failure_handoff,
    finalize_watchdog_handoff,
)
from tdcsim_cbo.verifier import verify_scenario_run


_CLAIM_NAME = ".tdcsim-cbo-bounded-writer.claim"


def test_watchdog_progress_emits_flushed_thirty_period_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "run"
    staging = tmp_path / ".run.staging-123-test"
    claim = tmp_path / _CLAIM_NAME
    progress_path = tmp_path / ".run.watchdog-handoff-test.progress.json"
    printed: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        "builtins.print",
        lambda *args, **kwargs: printed.append((args, kwargs)),
    )
    progress = {
        "progress_state": "period_complete",
        "admission_date": None,
        "last_completed_period": "2026-07-19",
        "period_count": 29,
        "event_count": 290,
        "event_root_sha256": "a" * 64,
        "peak_rss_bytes": 123,
        "failure_invariant": None,
        "failure_key": None,
        "last_events": [],
    }

    runner._record_watchdog_progress(
        progress_path,
        out=output,
        staging=staging,
        claim_path=claim,
        progress=progress,
    )
    assert printed == []

    checkpoint = {
        **progress,
        "last_completed_period": "2026-07-20",
        "period_count": 30,
        "event_count": 300,
        "peak_rss_bytes": 456,
    }
    runner._record_watchdog_progress(
        progress_path,
        out=output,
        staging=staging,
        claim_path=claim,
        progress=checkpoint,
    )

    assert printed == [
        (
            (
                "tdcsim: period_checkpoint "
                '{"event_count":300,'
                '"last_completed_period":"2026-07-20",'
                '"peak_rss_bytes":456,"period_count":30}',
            ),
            {"flush": True},
        )
    ]
    assert read_json(progress_path)["progress"] == checkpoint


def test_atomic_json_write_retries_transient_permission_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "progress.json"
    real_replace = runner.os.replace
    attempts = 0

    def transient_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient synced-file reader")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", transient_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    runner._write_json_atomic(path, {"status": "ok"})

    assert attempts == 3
    assert read_json(path) == {"status": "ok"}


def test_atomic_json_write_fails_after_bounded_permission_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "progress.json"
    attempts = 0

    def persistent_conflict(_source: Path, _destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        raise PermissionError("persistent synced-file reader")

    monkeypatch.setattr(runner.os, "replace", persistent_conflict)
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)

    with pytest.raises(PermissionError, match="persistent synced-file reader"):
        runner._write_json_atomic(path, {"status": "not-written"})

    assert attempts == runner._ATOMIC_REPLACE_MAX_ATTEMPTS
    assert not path.exists()


def _lightweight_limits() -> BoundedResourceLimits:
    """Keep tests independent of host memory while retaining explicit budgets."""

    high_watermark = 1 << 60
    return BoundedResourceLimits(
        minimum_available_bytes=0,
        application_abort_rss_bytes=high_watermark,
        parent_graceful_stop_rss_bytes=high_watermark,
        parent_kill_rss_bytes=high_watermark,
        acceptance_peak_rss_bytes=high_watermark,
        portfolio_row_budget=100_000,
        key_cardinality_budget=100_000,
    )


def _watchdog_limits() -> BoundedResourceLimits:
    return BoundedResourceLimits(
        minimum_available_bytes=0,
        portfolio_row_budget=100_000,
        key_cardinality_budget=100_000,
    )


def _watchdog_result(
    *,
    peak_rss_bytes: int,
    returncode: int = 0,
    action: str = "completed",
) -> WatchdogResult:
    return WatchdogResult(
        child_pid=os.getpid(),
        returncode=returncode,
        action=action,
        peak_rss_bytes=peak_rss_bytes,
        last_rss_bytes=peak_rss_bytes,
        acceptance_peak_rss_bytes=6 * GIB,
        terminate_rss_bytes=10 * GIB,
        kill_rss_bytes=12 * GIB,
        poll_interval_seconds=1.0,
    )


def _run_scoped_artifacts(parent: Path, output_name: str) -> list[Path]:
    return sorted(
        path
        for path in parent.iterdir()
        if path.name == output_name
        or path.name.startswith(f".{output_name}.staging-")
        or path.name.startswith(f"{output_name}.failure-")
    )


def _inject_resource_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the bounded sink's host probes deterministic for protocol tests."""

    production_sink = runner.BoundedScenarioEvidenceSink

    def deterministic_sink(output_dir, *, limits, progress_callback=None):
        return production_sink(
            output_dir,
            limits=limits,
            rss_reader=lambda: 64 * 1024 * 1024,
            available_reader=lambda: 256 * 1024 * 1024 * 1024,
            cpu_reader=lambda: 1.0,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(runner, "BoundedScenarioEvidenceSink", deterministic_sink)
    for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        monkeypatch.setenv(name, "1")


def test_live_campaign_claim_blocks_a_second_scenario_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production acquisition path must exclude a nested second writer."""

    claim_path = tmp_path / _CLAIM_NAME
    first_output = tmp_path / "first-scenario"
    second_output = tmp_path / "second-scenario"
    first_spec = SimpleNamespace(scenario_id="first")
    second_spec = SimpleNamespace(scenario_id="second")
    second_writer_was_blocked = False

    class StopFirstWriter(RuntimeError):
        pass

    def inspect_live_claim_then_stop(self, baseline, spec, output_dir):
        nonlocal second_writer_was_blocked
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        assert claim["pid"] == os.getpid()
        assert claim["scenario_id"] == "first"
        assert claim["output_dir"] == first_output.name
        with pytest.raises(RunnerError, match="already claimed"):
            runner.run_cbo_scenario(
                object(),
                second_spec,
                second_output,
                resource_limits=_lightweight_limits(),
            )
        second_writer_was_blocked = True
        raise StopFirstWriter("end lightweight first-writer probe")

    monkeypatch.setattr(
        runner.CboScenarioCompiler,
        "compile",
        inspect_live_claim_then_stop,
    )

    with pytest.raises(StopFirstWriter, match="first-writer probe"):
        runner.run_cbo_scenario(
            object(),
            first_spec,
            first_output,
            resource_limits=_lightweight_limits(),
        )

    assert second_writer_was_blocked is True
    assert not second_output.exists()
    assert not claim_path.exists()


@pytest.mark.parametrize(
    "claim_bytes",
    [
        (
            b'{"claimed_at_utc":"1970-01-01T00:00:00+00:00",'
            b'"output_dir":"old-run","pid":2147483647,"scenario_id":"old"}'
        ),
        b"owner liveness is unknown; preserve this evidence",
    ],
    ids=["apparently-stale", "unknown-owner"],
)
def test_existing_stale_or_unknown_claim_is_preserved_and_blocks(
    tmp_path: Path,
    claim_bytes: bytes,
) -> None:
    """The runner must not guess that an existing campaign claim is reclaimable."""

    claim_path = tmp_path / _CLAIM_NAME
    claim_path.write_bytes(claim_bytes)
    output = tmp_path / "blocked-scenario"

    with pytest.raises(RunnerError, match="already claimed"):
        runner.run_cbo_scenario(
            object(),
            SimpleNamespace(scenario_id="blocked"),
            output,
            resource_limits=_lightweight_limits(),
        )

    assert claim_path.read_bytes() == claim_bytes
    assert not output.exists()
    assert _run_scoped_artifacts(tmp_path, output.name) == []


def test_success_removes_claim_and_promotes_only_a_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a staged tree with a complete manifest may become the final run."""

    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    spec = CboScenarioSpec.from_file(scenarios["noop"])
    output = tmp_path / "successful-run"
    claim_path = tmp_path / _CLAIM_NAME
    staged_path: Path | None = None
    _inject_resource_readers(monkeypatch)

    def inspect_before_promotion(root: Path) -> None:
        nonlocal staged_path
        staged_path = root
        assert root.parent == output.parent
        assert root.name.startswith(f".{output.name}.staging-")
        assert not output.exists()
        assert claim_path.exists()
        assert read_json(root / "tdcsim_cbo_run_manifest.json")["status"] == "complete"

    monkeypatch.setattr(runner, "_fsync_run_tree", inspect_before_promotion)

    completed = runner.run_cbo_scenario(
        baseline,
        spec,
        output,
        resource_limits=_lightweight_limits(),
    )

    assert staged_path is not None
    assert not staged_path.exists()
    assert completed.output_dir == output
    assert read_json(completed.manifest_path)["status"] == "complete"
    assert completed.results_path.exists()
    assert not claim_path.exists()
    assert not list(tmp_path.glob(f"{output.name}.failure-*.json"))
    assert _run_scoped_artifacts(tmp_path, output.name) == [output]
    local_verification = verify_scenario_run(output)
    replay_verification = verify_scenario_run(
        output,
        baseline_package=baseline.package_path,
        attestation=baseline.attestation.path,
    )
    assert local_verification["status"] == "pass"
    assert local_verification["verification_grade"] == "bounded_replay_v1"
    assert replay_verification["status"] == "pass"
    assert replay_verification["verification_grade"] == "bounded_replay_v1"


def test_injected_failure_leaves_only_a_nonpromotable_sibling_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after sink creation must discard staging and never publish output."""

    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    spec = CboScenarioSpec.from_file(scenarios["noop"])
    output = tmp_path / "failed-run"
    claim_path = tmp_path / _CLAIM_NAME
    _inject_resource_readers(monkeypatch)

    class InjectedSimulationFailure(RuntimeError):
        pass

    def fail_before_simulation(*args, **kwargs):
        raise InjectedSimulationFailure("injected bounded-run failure")

    monkeypatch.setattr(runner, "run_simulation", fail_before_simulation)

    with pytest.raises(InjectedSimulationFailure, match="bounded-run failure"):
        runner.run_cbo_scenario(
            baseline,
            spec,
            output,
            resource_limits=_lightweight_limits(),
        )

    receipts = list(tmp_path.glob(f"{output.name}.failure-*.json"))
    assert len(receipts) == 1
    failure = read_json(receipts[0])
    assert failure["status"] == "failed"
    assert failure["exception_class"] == "InjectedSimulationFailure"
    assert failure["promotable"] is False
    assert failure["output_dir_name"] == output.name
    assert not output.exists()
    assert not claim_path.exists()
    assert _run_scoped_artifacts(tmp_path, output.name) == receipts


def test_mid_period_writer_failure_cannot_publish_a_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    spec = CboScenarioSpec.from_file(scenarios["noop"])
    output = tmp_path / "writer-failed-run"
    claim_path = tmp_path / _CLAIM_NAME
    _inject_resource_readers(monkeypatch)
    original_write_rows = bounded_output._DeterministicCsvGzip.write_rows

    class InjectedWriterFailure(OSError):
        pass

    def fail_ledger_writer(self, rows):
        if self.path.name == "tdcsim_period_ledger_totals.csv.gz" and rows:
            raise InjectedWriterFailure("injected compact-ledger writer failure")
        return original_write_rows(self, rows)

    monkeypatch.setattr(
        bounded_output._DeterministicCsvGzip,
        "write_rows",
        fail_ledger_writer,
    )

    with pytest.raises(InjectedWriterFailure, match="ledger writer"):
        runner.run_cbo_scenario(
            baseline,
            spec,
            output,
            resource_limits=_lightweight_limits(),
        )

    receipts = list(tmp_path.glob(f"{output.name}.failure-*.json"))
    assert len(receipts) == 1
    failure = read_json(receipts[0])
    assert failure["status"] == "failed"
    assert failure["exception_class"] == "InjectedWriterFailure"
    assert failure["promotable"] is False
    assert not output.exists()
    assert not claim_path.exists()
    assert _run_scoped_artifacts(tmp_path, output.name) == receipts


def test_parent_watchdog_owns_final_promotion_and_records_full_child_peak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    spec = CboScenarioSpec.from_file(scenarios["noop"])
    output = tmp_path / "watchdog-accepted-run"
    handoff = tmp_path / f".{output.name}.watchdog-handoff-test.json"
    claim_path = tmp_path / _CLAIM_NAME
    _inject_resource_readers(monkeypatch)

    prepared = runner.run_cbo_scenario(
        baseline,
        spec,
        output,
        resource_limits=_watchdog_limits(),
        watchdog_handoff=handoff,
    )

    assert not output.exists()
    assert prepared.output_dir.name.startswith(f".{output.name}.staging-")
    assert prepared.output_dir.exists()
    assert prepared.run_manifest["status"] == "pending_parent_watchdog_acceptance"
    assert not (prepared.output_dir / "tdcsim_cbo_run_manifest.json").exists()
    assert handoff.exists()
    assert claim_path.exists()
    progress_path = handoff.with_name(f"{handoff.stem}.progress.json")
    progress = read_json(progress_path)
    assert progress["progress"]["progress_state"] == "period_complete"
    assert progress["progress"]["period_count"] > 0
    assert progress["progress"]["event_count"] > 0

    manifest = finalize_watchdog_handoff(
        output,
        handoff,
        _watchdog_result(peak_rss_bytes=5 * GIB),
    )

    assert manifest is not None
    assert output.exists()
    assert not prepared.output_dir.exists()
    assert not handoff.exists()
    assert not progress_path.exists()
    assert not claim_path.exists()
    recorded = read_json(output / "tdcsim_cbo_run_manifest.json")
    assert recorded["status"] == "complete"
    assert recorded["parent_watchdog"] == {
        "status": "accepted",
        "sampler": "parent_process_rss_poll_v1",
        "child_pid": os.getpid(),
        "child_returncode": 0,
        "action": "completed",
        "peak_rss_bytes": 5 * GIB,
        "worker_peak_rss_bytes": 64 * 1024**2,
        "effective_peak_rss_bytes": 5 * GIB,
        "acceptance_peak_rss_bytes": 6 * GIB,
        "terminate_rss_bytes": 10 * GIB,
        "kill_rss_bytes": 12 * GIB,
        "poll_interval_seconds": 1.0,
    }
    assert {
        item["id"]: item for item in recorded["validation"]["invariants"]
    }["parent_watchdog_peak_rss"] == {
        "id": "parent_watchdog_peak_rss",
        "status": "pass",
        "observed": 5 * GIB,
        "limit": 6 * GIB,
    }
    assert verify_scenario_run(output)["verification_grade"] == (
        "bounded_replay_v1"
    )


def test_parent_watchdog_rejects_peak_above_six_gib_without_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    spec = CboScenarioSpec.from_file(scenarios["noop"])
    output = tmp_path / "watchdog-rejected-run"
    handoff = tmp_path / f".{output.name}.watchdog-handoff-test.json"
    claim_path = tmp_path / _CLAIM_NAME
    _inject_resource_readers(monkeypatch)

    prepared = runner.run_cbo_scenario(
        baseline,
        spec,
        output,
        resource_limits=_watchdog_limits(),
        watchdog_handoff=handoff,
    )
    accepted = finalize_watchdog_handoff(
        output,
        handoff,
        _watchdog_result(peak_rss_bytes=6 * GIB + 1),
    )

    assert accepted is None
    assert not output.exists()
    assert not prepared.output_dir.exists()
    assert not handoff.exists()
    assert not claim_path.exists()
    receipts = list(tmp_path.glob(f"{output.name}.watchdog-failure-*.json"))
    assert len(receipts) == 1
    receipt = read_json(receipts[0])
    assert receipt["action"] == "reject_acceptance_peak"
    assert receipt["peak_rss_bytes"] == 6 * GIB + 1
    assert receipt["acceptance_peak_rss_bytes"] == 6 * GIB


def test_natural_worker_failure_handoff_preserves_bounded_progress_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline, scenarios = cbo_fixtures._runner_baseline_and_scenarios(tmp_path)
    spec = CboScenarioSpec.from_file(scenarios["noop"])
    output = tmp_path / "watchdog-worker-failed-run"
    handoff = tmp_path / f".{output.name}.watchdog-handoff-test.json"
    _inject_resource_readers(monkeypatch)

    class InjectedSimulationFailure(RuntimeError):
        invariant_id = "cash_chain_closure"
        failure_key = "TGA"

    def fail_before_simulation(*args, **kwargs):
        raise InjectedSimulationFailure("injected bounded-run failure")

    monkeypatch.setattr(runner, "run_simulation", fail_before_simulation)
    with pytest.raises(InjectedSimulationFailure, match="bounded-run failure"):
        runner.run_cbo_scenario(
            baseline,
            spec,
            output,
            resource_limits=_watchdog_limits(),
            watchdog_handoff=handoff,
        )

    assert not output.exists()
    assert handoff.exists()
    progress_path = handoff.with_name(f"{handoff.stem}.progress.json")
    assert progress_path.exists()
    assert not list(tmp_path.glob(f"{output.name}.failure-*.json"))
    failure = consume_watchdog_failure_handoff(
        output,
        handoff,
        _watchdog_result(peak_rss_bytes=128 * 1024**2, returncode=1),
    )
    assert not handoff.exists()
    assert not progress_path.exists()
    assert failure["last_completed_period"] is None
    assert failure["event_count"] == 0
    assert len(failure["event_root_sha256"]) == 64
    assert failure["failure_invariant"] == "cash_chain_closure"
    assert failure["failure_key"] == "TGA"
    assert failure["last_events"] == []


def test_forced_intervention_cleans_only_owned_artifacts_and_returns_progress(
    tmp_path: Path,
) -> None:
    output = tmp_path / "forced-run"
    handoff = tmp_path / f".{output.name}.watchdog-handoff-test.json"
    progress_path = handoff.with_name(f"{handoff.stem}.progress.json")
    claim_path = tmp_path / _CLAIM_NAME
    owned_staging = tmp_path / f".{output.name}.staging-{os.getpid()}-0123456789"
    unrelated_staging = (
        tmp_path / f".{output.name}.staging-{os.getpid() + 1}-0123456789"
    )
    owned_staging.mkdir()
    unrelated_staging.mkdir()
    handoff.write_text("parent-owned control\n", encoding="utf-8")
    handoff_temp = tmp_path / f".{handoff.name}.interrupted.tmp"
    progress_temp = tmp_path / f".{progress_path.name}.interrupted.tmp"
    handoff_temp.write_text("partial", encoding="utf-8")
    progress_temp.write_text("partial", encoding="utf-8")
    claim_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "scenario_id": "forced",
                "output_dir": output.name,
            }
        ),
        encoding="utf-8",
    )
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": "tdcsim_cbo_watchdog_progress_v1",
                "worker_pid": os.getpid(),
                "output_dir_name": output.name,
                "staging_dir_name": owned_staging.name,
                "claim_file_name": _CLAIM_NAME,
                "progress": {
                    "progress_state": "period_complete",
                    "admission_date": None,
                    "last_completed_period": "2032-09-30",
                    "period_count": 9,
                    "event_count": 123,
                    "event_root_sha256": "c" * 64,
                    "peak_rss_bytes": 5 * GIB,
                    "last_events": [{"event_seq": 123}],
                },
            }
        ),
        encoding="utf-8",
    )

    details = cleanup_watchdog_intervention(
        output,
        handoff,
        _watchdog_result(
            peak_rss_bytes=10 * GIB,
            returncode=-15,
            action="terminate_rss",
        ),
    )

    assert details["last_completed_period"] == "2032-09-30"
    assert details["event_count"] == 123
    assert details["forced_cleanup"]["ownership_proved"] is True
    assert details["forced_cleanup"]["claim_removed"] is True
    assert details["forced_cleanup"]["staging_dirs_removed"] == [
        owned_staging.name
    ]
    assert not owned_staging.exists()
    assert unrelated_staging.exists()
    assert not claim_path.exists()
    assert not handoff.exists()
    assert not progress_path.exists()
    assert not handoff_temp.exists()
    assert not progress_temp.exists()


def test_monitor_error_cleanup_uses_exact_claim_when_progress_is_absent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "monitor-error-run"
    handoff = tmp_path / f".{output.name}.watchdog-handoff-test.json"
    progress_path = handoff.with_name(f"{handoff.stem}.progress.json")
    claim_path = tmp_path / _CLAIM_NAME
    owned_staging = tmp_path / f".{output.name}.staging-{os.getpid()}-abcdef0123"
    owned_staging.mkdir()
    handoff.write_text("parent-owned control\n", encoding="utf-8")
    claim_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "scenario_id": "monitor-error",
                "output_dir": output.name,
            }
        ),
        encoding="utf-8",
    )

    details = cleanup_watchdog_intervention(
        output,
        handoff,
        _watchdog_result(
            peak_rss_bytes=0,
            returncode=-15,
            action="monitor_error_terminate",
        ),
    )

    assert details["forced_cleanup"] == {
        "ownership_proved": True,
        "claim_removed": True,
        "staging_dirs_removed": [owned_staging.name],
        "progress_valid": False,
    }
    assert not owned_staging.exists()
    assert not claim_path.exists()
    assert not handoff.exists()
    assert not progress_path.exists()


def test_forced_cleanup_preserves_every_artifact_when_pid_ownership_mismatches(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mismatched-owner-run"
    handoff = tmp_path / f".{output.name}.watchdog-handoff-test.json"
    progress_path = handoff.with_name(f"{handoff.stem}.progress.json")
    claim_path = tmp_path / _CLAIM_NAME
    actual_pid = os.getpid() + 100
    watched_pid = os.getpid()
    staging = tmp_path / f".{output.name}.staging-{actual_pid}-abcdef0123"
    staging.mkdir()
    handoff.write_text("preserve parent control\n", encoding="utf-8")
    handoff_temp = tmp_path / f".{handoff.name}.interrupted.tmp"
    progress_temp = tmp_path / f".{progress_path.name}.interrupted.tmp"
    handoff_temp.write_text("preserve", encoding="utf-8")
    progress_temp.write_text("preserve", encoding="utf-8")
    claim_path.write_text(
        json.dumps(
            {
                "pid": actual_pid,
                "scenario_id": "mismatched-owner",
                "output_dir": output.name,
            }
        ),
        encoding="utf-8",
    )
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": "tdcsim_cbo_watchdog_progress_v1",
                "worker_pid": actual_pid,
                "output_dir_name": output.name,
                "staging_dir_name": staging.name,
                "claim_file_name": _CLAIM_NAME,
                "progress": {
                    "progress_state": "period_complete",
                    "admission_date": None,
                    "last_completed_period": "2036-09-30",
                    "period_count": 3_754,
                    "event_count": 29_215_887,
                    "event_root_sha256": "d" * 64,
                    "peak_rss_bytes": 376_762_368,
                    "last_events": [],
                },
            }
        ),
        encoding="utf-8",
    )

    details = cleanup_watchdog_intervention(
        output,
        handoff,
        WatchdogResult(
            child_pid=watched_pid,
            returncode=120,
            action="child_exit_nonzero",
            peak_rss_bytes=6_000_000,
            last_rss_bytes=6_000_000,
            acceptance_peak_rss_bytes=6 * GIB,
            terminate_rss_bytes=10 * GIB,
            kill_rss_bytes=12 * GIB,
            poll_interval_seconds=1.0,
        ),
    )

    assert details["forced_cleanup"]["ownership_proved"] is False
    assert details["forced_cleanup"]["claim_removed"] is False
    assert details["forced_cleanup"]["staging_dirs_removed"] == []
    assert details["forced_cleanup"]["progress_valid"] is False
    assert details["forced_cleanup"]["progress_error"] == "RunnerError"
    assert staging.exists()
    assert claim_path.exists()
    assert handoff.exists()
    assert progress_path.exists()
    assert handoff_temp.exists()
    assert progress_temp.exists()
