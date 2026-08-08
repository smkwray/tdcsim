from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

import tdcsim_cbo.cli as cli
from scripts import run_cbo_scenario as legacy_run_script
from tdcsim_cbo.process_watchdog import (
    DEFAULT_ACCEPTANCE_PEAK_RSS_BYTES,
    DEFAULT_KILL_RSS_BYTES,
    DEFAULT_TERMINATE_RSS_BYTES,
    GIB,
    ProcessRssError,
    THREAD_LIMIT_ENVIRONMENT_VARIABLES,
    WatchdogLimits,
    WatchdogResult,
    monitor_process,
    process_rss_bytes,
    run_command_with_watchdog,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProcess:
    def __init__(
        self,
        *,
        pid: int = 4321,
        polls_before_exit: int | None = None,
        natural_returncode: int = 0,
        exit_on_terminate: bool = True,
    ) -> None:
        self.pid = pid
        self.polls_before_exit = polls_before_exit
        self.natural_returncode = natural_returncode
        self.exit_on_terminate = exit_on_terminate
        self.poll_calls = 0
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        self.poll_calls += 1
        if (
            self.polls_before_exit is not None
            and self.poll_calls > self.polls_before_exit
        ):
            self.returncode = self.natural_returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.returncode = -15

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode


class SequenceSampler:
    def __init__(self, values: Sequence[int | None]) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self, _pid: int) -> int | None:
        if not self.values:
            raise AssertionError("sampler requires at least one value")
        value = self.values[min(self.index, len(self.values) - 1)]
        self.index += 1
        return value


def test_default_watchdog_thresholds_match_memory_design() -> None:
    limits = WatchdogLimits()

    assert DEFAULT_TERMINATE_RSS_BYTES == 10 * GIB
    assert DEFAULT_KILL_RSS_BYTES == 12 * GIB
    assert DEFAULT_ACCEPTANCE_PEAK_RSS_BYTES == 6 * GIB
    assert limits.acceptance_peak_rss_bytes == 6 * GIB
    assert limits.terminate_rss_bytes == 10 * GIB
    assert limits.kill_rss_bytes == 12 * GIB


def test_watchdog_records_peak_and_allows_a_bounded_worker_to_finish() -> None:
    process = FakeProcess(polls_before_exit=3)
    clock = FakeClock()
    samples: list[tuple[int, int, int]] = []

    result = monitor_process(
        process,
        sample_rss=SequenceSampler([1 * GIB, 3 * GIB, 2 * GIB]),
        sample_callback=lambda pid, rss, peak: samples.append(
            (pid, rss, peak)
        ),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.action == "completed"
    assert result.returncode == 0
    assert result.peak_rss_bytes == 3 * GIB
    assert result.last_rss_bytes == 2 * GIB
    assert process.terminate_calls == 0
    assert process.kill_calls == 0
    assert samples == [
        (process.pid, 1 * GIB, 1 * GIB),
        (process.pid, 3 * GIB, 3 * GIB),
        (process.pid, 2 * GIB, 3 * GIB),
    ]


def test_watchdog_terminates_at_the_exact_ten_gib_boundary() -> None:
    process = FakeProcess()
    clock = FakeClock()

    result = monitor_process(
        process,
        sample_rss=SequenceSampler([10 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.action == "terminate_rss"
    assert result.peak_rss_bytes == 10 * GIB
    assert process.terminate_calls == 1
    assert process.kill_calls == 0


def test_watchdog_kills_at_the_exact_twelve_gib_boundary() -> None:
    process = FakeProcess()
    clock = FakeClock()

    result = monitor_process(
        process,
        sample_rss=SequenceSampler([12 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert result.action == "kill_rss"
    assert result.peak_rss_bytes == 12 * GIB
    assert process.terminate_calls == 0
    assert process.kill_calls == 1


def test_watchdog_kills_a_worker_that_ignores_termination_after_grace() -> None:
    process = FakeProcess(exit_on_terminate=False)
    clock = FakeClock()
    limits = WatchdogLimits(
        poll_interval_seconds=1.0,
        terminate_grace_seconds=2.0,
    )

    result = monitor_process(
        process,
        sample_rss=SequenceSampler([10 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        limits=limits,
    )

    assert result.action == "kill_after_grace"
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert clock.now == pytest.approx(2.0)


def test_live_worker_with_missing_rss_fails_closed() -> None:
    process = FakeProcess()

    with pytest.raises(ProcessRssError, match="no value"):
        monitor_process(
            process,
            sample_rss=SequenceSampler([None]),
            monotonic=lambda: 0.0,
            sleep=lambda _seconds: None,
        )


def test_watchdog_wrapper_writes_sibling_receipt_on_intervention(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    clock = FakeClock()
    captured: list[list[str]] = []

    def fake_popen(command: Sequence[str]) -> FakeProcess:
        captured.append(list(command))
        return process

    output_dir = tmp_path / "requested-run"
    status = run_command_with_watchdog(
        ["python", "-m", "worker"],
        output_dir=output_dir,
        popen_factory=fake_popen,
        sample_rss=SequenceSampler([10 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        intervention_details_callback=lambda _result: {
            "progress_state": "period_complete",
            "last_completed_period": "2031-09-30",
            "event_count": 41,
            "event_root_sha256": "b" * 64,
            "last_events": [{"event_seq": 41}],
        },
    )

    assert status == 1
    assert captured == [["python", "-m", "worker"]]
    assert not output_dir.exists()
    receipts = list(tmp_path.glob("requested-run.watchdog-failure-*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["failure_kind"] == "parent_memory_watchdog"
    assert payload["action"] == "terminate_rss"
    assert payload["peak_rss_bytes"] == 10 * GIB
    assert payload["terminate_rss_bytes"] == 10 * GIB
    assert payload["kill_rss_bytes"] == 12 * GIB
    assert payload["worker_failure"]["last_completed_period"] == "2031-09-30"
    assert payload["worker_failure"]["event_count"] == 41


def test_monitor_error_cleans_stopped_child_and_writes_one_enriched_receipt(
    tmp_path: Path,
) -> None:
    process = FakeProcess()
    cleaned: list[object] = []

    def cleanup(result):
        assert process.poll() == -15
        cleaned.append(result)
        return {
            "progress_state": "admission",
            "last_completed_period": None,
            "event_count": 0,
            "event_root_sha256": "0" * 64,
            "last_events": [],
        }

    output_dir = tmp_path / "requested-run"
    status = run_command_with_watchdog(
        ["python", "-m", "worker"],
        output_dir=output_dir,
        popen_factory=lambda _command: process,
        sample_rss=SequenceSampler([None]),
        monotonic=lambda: 0.0,
        sleep=lambda _seconds: None,
        intervention_details_callback=cleanup,
    )

    assert status == 1
    assert len(cleaned) == 1
    receipts = list(tmp_path.glob("requested-run.watchdog-failure-*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["action"] == "monitor_error_terminate"
    assert payload["monitor_error"] == "ProcessRssError"
    assert payload["worker_failure"]["progress_state"] == "admission"


def test_watchdog_wrapper_records_one_enriched_natural_child_failure_receipt(
    tmp_path: Path,
) -> None:
    process = FakeProcess(polls_before_exit=1, natural_returncode=7)
    clock = FakeClock()

    status = run_command_with_watchdog(
        ["python", "-m", "worker"],
        output_dir=tmp_path / "requested-run",
        popen_factory=lambda _command: process,
        sample_rss=SequenceSampler([1 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        failure_details_callback=lambda _result: {
            "last_completed_period": "2030-09-30",
            "event_count": 17,
            "event_root_sha256": "a" * 64,
            "failure_invariant": "cash_chain_closure",
            "failure_key": "TGA",
            "last_events": [{"event_seq": 17, "event_type": "tax"}],
        },
    )

    assert status == 7
    receipts = list(tmp_path.glob("requested-run.watchdog-failure-*.json"))
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["failure_kind"] == "worker_failure"
    assert payload["action"] == "child_exit_nonzero"
    assert payload["worker_failure"]["last_completed_period"] == "2030-09-30"
    assert payload["worker_failure"]["event_count"] == 17
    assert payload["worker_failure"]["event_root_sha256"] == "a" * 64
    assert payload["worker_failure"]["failure_invariant"] == "cash_chain_closure"
    assert payload["worker_failure"]["failure_key"] == "TGA"
    assert payload["worker_failure"]["last_events"] == [
        {"event_seq": 17, "event_type": "tax"}
    ]


def test_completed_worker_above_six_gib_requires_parent_rejection(
    tmp_path: Path,
) -> None:
    process = FakeProcess(polls_before_exit=1)
    clock = FakeClock()
    observed = []

    status = run_command_with_watchdog(
        ["python", "-m", "worker"],
        output_dir=tmp_path / "requested-run",
        popen_factory=lambda _command: process,
        sample_rss=SequenceSampler([7 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        completion_callback=lambda result: observed.append(result) is None and False,
    )

    assert status == 1
    assert len(observed) == 1
    assert observed[0].peak_rss_bytes == 7 * GIB
    assert observed[0].acceptance_peak_rss_bytes == 6 * GIB


def test_acceptance_callback_error_runs_cleanup_before_failure_receipt(
    tmp_path: Path,
) -> None:
    process = FakeProcess(polls_before_exit=1)
    clock = FakeClock()
    cleaned: list[WatchdogResult] = []

    def fail_acceptance(_result: WatchdogResult) -> bool:
        raise ValueError("invalid staged lineage")

    def cleanup(result: WatchdogResult) -> dict[str, object]:
        cleaned.append(result)
        return {
            "progress_state": "period_complete",
            "last_completed_period": "2036-09-30",
            "event_count": 101,
        }

    output = tmp_path / "requested-run"
    status = run_command_with_watchdog(
        ["python", "-m", "worker"],
        output_dir=output,
        popen_factory=lambda _command: process,
        sample_rss=SequenceSampler([1 * GIB]),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        completion_callback=fail_acceptance,
        intervention_details_callback=cleanup,
    )

    assert status == 1
    assert len(cleaned) == 1
    receipts = list(
        tmp_path.glob("requested-run.watchdog-failure-*.json")
    )
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert payload["action"] == "acceptance_callback_error"
    assert payload["monitor_error"] == "ValueError"
    assert payload["worker_failure"]["event_count"] == 101


def test_cli_run_uses_hidden_worker_process_before_opening_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    launcher_was_present = cli._PYVENV_LAUNCHER_ENV in os.environ
    prior_launcher = os.environ.get(cli._PYVENV_LAUNCHER_ENV)
    prior_thread_environment = {
        name: os.environ.get(name)
        for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    }

    def fake_watchdog(
        command: Sequence[str],
        *,
        output_dir: str | Path,
        popen_factory,
        sample_callback,
        completion_callback,
        failure_details_callback,
        intervention_details_callback,
    ) -> int:
        captured["command"] = list(command)
        captured["output_dir"] = Path(output_dir)
        captured["completion_callback"] = completion_callback
        captured["failure_details_callback"] = failure_details_callback
        captured["intervention_details_callback"] = intervention_details_callback
        captured["popen_factory"] = popen_factory
        captured["sample_callback"] = sample_callback
        captured["venv_launcher"] = os.environ.get(
            cli._PYVENV_LAUNCHER_ENV
        )
        captured["thread_environment"] = {
            name: os.environ.get(name)
            for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
        }
        child = popen_factory(
            [
                command[0],
                "-B",
                "-c",
                (
                    "import sys;"
                    "print('durable-worker-stdout');"
                    "print('durable-worker-stderr',file=sys.stderr)"
                ),
            ]
        )
        assert child.wait(timeout=30) == 0
        sample_callback(child.pid, 123, 456)
        captured["failure_details"] = failure_details_callback(
            WatchdogResult(
                child_pid=child.pid,
                returncode=120,
                action="completed",
                peak_rss_bytes=456,
                last_rss_bytes=123,
                acceptance_peak_rss_bytes=6 * GIB,
                terminate_rss_bytes=10 * GIB,
                kill_rss_bytes=12 * GIB,
                poll_interval_seconds=1.0,
            )
        )
        return 23

    monkeypatch.setattr(cli, "run_command_with_watchdog", fake_watchdog)
    output_dir = tmp_path / "new" / "nested" / "run"
    status = cli.main(
        [
            "run",
            "--baseline",
            str(tmp_path / "missing-baseline.zip"),
            "--attestation",
            str(tmp_path / "missing-attestation.json"),
            "--scenario",
            str(tmp_path / "missing-scenario.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    command = captured["command"]
    assert isinstance(command, list)
    worker_executable, venv_launcher = cli._watchdog_worker_launch()
    assert command[:3] == [worker_executable, "-m", "tdcsim_cbo.cli"]
    assert captured["venv_launcher"] == (
        venv_launcher if venv_launcher is not None else prior_launcher
    )
    assert "--_watchdog-worker" in command
    handoff_index = command.index("--_watchdog-handoff")
    assert Path(command[handoff_index + 1]).parent == output_dir.parent
    assert Path(command[handoff_index + 1]).name.startswith(
        f".{output_dir.name}.watchdog-handoff-"
    )
    assert captured["output_dir"] == output_dir.resolve()
    assert output_dir.parent.is_dir()
    assert callable(captured["completion_callback"])
    assert callable(captured["failure_details_callback"])
    assert callable(captured["intervention_details_callback"])
    assert callable(captured["popen_factory"])
    assert callable(captured["sample_callback"])
    failure_details = captured["failure_details"]
    assert isinstance(failure_details, dict)
    worker_log = failure_details["worker_standard_stream_log"]
    assert isinstance(worker_log, dict)
    worker_log_path = output_dir.parent / worker_log["file_name"]
    assert sorted(
        worker_log_path.read_text(encoding="utf-8").splitlines()
    ) == [
        "durable-worker-stderr",
        "durable-worker-stdout",
    ]
    assert worker_log["bytes"] == worker_log_path.stat().st_size
    assert worker_log["sha256"] == cli.sha256_file(worker_log_path)
    assert failure_details["watchdog_handoff_status"] == "unavailable"
    assert failure_details["watchdog_handoff_error"] == "RunnerError"
    assert captured["thread_environment"] == {
        name: "1" for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    }
    assert {
        name: os.environ.get(name)
        for name in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    } == prior_thread_environment
    assert (cli._PYVENV_LAUNCHER_ENV in os.environ) is launcher_was_present
    assert os.environ.get(cli._PYVENV_LAUNCHER_ENV) == prior_launcher
    assert status == 23


def test_controller_message_stream_failure_never_stops_watchdog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosedStream:
        def write(self, _value: object) -> None:
            raise ValueError("I/O operation on closed file")

        def flush(self) -> None:
            raise ValueError("I/O operation on closed file")

        def fileno(self) -> int:
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(cli.sys, "stdout", ClosedStream())

    cli._emit_watchdog_controller_message("connection already closed")


@pytest.mark.skipif(
    platform.system() != "Windows" or sys.prefix == sys.base_prefix,
    reason="requires the Windows CPython venv redirector",
)
def test_windows_watchdog_launch_binds_popen_to_real_venv_interpreter() -> None:
    worker_executable, venv_launcher = cli._watchdog_worker_launch()
    assert venv_launcher == sys.executable
    environment = os.environ.copy()
    environment[cli._PYVENV_LAUNCHER_ENV] = venv_launcher
    child_code = (
        "import json,os,sys;"
        "print(json.dumps({"
        "'pid':os.getpid(),"
        "'executable':sys.executable,"
        "'prefix':sys.prefix"
        "}))"
    )
    process = subprocess.Popen(
        [worker_executable, "-B", "-c", child_code],
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    stdout, _ = process.communicate(timeout=30)
    child = json.loads(stdout)

    assert process.returncode == 0
    assert process.pid == child["pid"]
    assert os.path.normcase(os.path.abspath(child["executable"])) == (
        os.path.normcase(os.path.abspath(sys.executable))
    )
    assert os.path.normcase(os.path.abspath(child["prefix"])) == (
        os.path.normcase(os.path.abspath(sys.prefix))
    )


def test_legacy_run_script_delegates_to_the_watched_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_argv = [
        "--baseline",
        str(tmp_path / "baseline.zip"),
        "--attestation",
        str(tmp_path / "attestation.json"),
        "--scenario",
        str(tmp_path / "scenario.json"),
        "--output-dir",
        str(tmp_path / "run"),
        "--profile",
        "compact",
    ]
    captured: list[list[str]] = []

    def fake_cli_main(argv: Sequence[str] | None = None) -> int:
        captured.append(list(argv or []))
        return 29

    monkeypatch.setattr(legacy_run_script.cbo_cli, "main", fake_cli_main)

    assert legacy_run_script.main(raw_argv) == 29
    assert captured == [["run", *raw_argv]]
    assert not hasattr(legacy_run_script, "run_cbo_scenario")


def test_hidden_worker_runs_the_library_path_without_reforking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = object()
    spec = SimpleNamespace(assert_baseline_matches=lambda value: value is baseline)
    handoff_path = tmp_path / ".run.watchdog-handoff-test.json"
    run = SimpleNamespace(
        run_manifest={"output_manifest": {"summary": {"sha256": "worker-summary"}}}
    )

    monkeypatch.setattr(
        cli.CboBaselinePackage,
        "open",
        lambda *_args, **_kwargs: baseline,
    )
    monkeypatch.setattr(
        cli.CboScenarioSpec,
        "from_file",
        lambda *_args, **_kwargs: spec,
    )
    monkeypatch.setattr(cli, "run_cbo_scenario", lambda *_args, **_kwargs: run)
    monkeypatch.setattr(
        cli,
        "run_command_with_watchdog",
        lambda *_args, **_kwargs: pytest.fail("worker recursively started a watchdog"),
    )

    status = cli.main(
        [
            "run",
            "--baseline",
            str(tmp_path / "baseline.zip"),
            "--attestation",
            str(tmp_path / "attestation.json"),
            "--scenario",
            str(tmp_path / "scenario.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--_watchdog-worker",
            "--_watchdog-handoff",
            str(handoff_path),
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == ""


def test_hidden_worker_flag_is_not_in_cli_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--_watchdog-worker" not in help_text
    assert "--_watchdog-handoff" not in help_text


def test_stdlib_sampler_reports_current_process_rss() -> None:
    rss = process_rss_bytes(os.getpid())

    assert isinstance(rss, int)
    assert rss > 0
