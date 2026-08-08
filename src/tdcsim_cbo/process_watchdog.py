"""Parent-process RSS watchdog for production CBO CLI runs.

The simulator remains an ordinary library call.  Only the CLI process boundary
uses this module: a small parent process starts one worker, samples that
worker's resident memory, and stops it if it crosses the declared ceilings.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


GIB = 1024**3
DEFAULT_ACCEPTANCE_PEAK_RSS_BYTES = 6 * GIB
DEFAULT_TERMINATE_RSS_BYTES = 10 * GIB
DEFAULT_KILL_RSS_BYTES = 12 * GIB
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_TERMINATE_GRACE_SECONDS = 10.0
DEFAULT_KILL_WAIT_SECONDS = 5.0
THREAD_LIMIT_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


class ProcessRssError(RuntimeError):
    """Raised when a live worker's RSS cannot be sampled safely."""


class WatchableProcess(Protocol):
    """The small subprocess surface used by :func:`monitor_process`."""

    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


@dataclass(frozen=True)
class WatchdogLimits:
    """Memory and timing limits for one production worker."""

    acceptance_peak_rss_bytes: int = DEFAULT_ACCEPTANCE_PEAK_RSS_BYTES
    terminate_rss_bytes: int = DEFAULT_TERMINATE_RSS_BYTES
    kill_rss_bytes: int = DEFAULT_KILL_RSS_BYTES
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    terminate_grace_seconds: float = DEFAULT_TERMINATE_GRACE_SECONDS
    kill_wait_seconds: float = DEFAULT_KILL_WAIT_SECONDS

    def __post_init__(self) -> None:
        if self.acceptance_peak_rss_bytes <= 0:
            raise ValueError("acceptance_peak_rss_bytes must be positive")
        if self.terminate_rss_bytes <= 0:
            raise ValueError("terminate_rss_bytes must be positive")
        if self.acceptance_peak_rss_bytes >= self.terminate_rss_bytes:
            raise ValueError(
                "acceptance_peak_rss_bytes must be below terminate_rss_bytes"
            )
        if self.kill_rss_bytes <= self.terminate_rss_bytes:
            raise ValueError("kill_rss_bytes must exceed terminate_rss_bytes")
        if self.poll_interval_seconds <= 0.0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.terminate_grace_seconds < 0.0:
            raise ValueError("terminate_grace_seconds must be nonnegative")
        if self.kill_wait_seconds <= 0.0:
            raise ValueError("kill_wait_seconds must be positive")


@dataclass(frozen=True)
class WatchdogResult:
    """Terminal observation for one monitored child."""

    child_pid: int
    returncode: int
    action: str
    peak_rss_bytes: int
    last_rss_bytes: int
    acceptance_peak_rss_bytes: int
    terminate_rss_bytes: int
    kill_rss_bytes: int
    poll_interval_seconds: float

    @property
    def intervened(self) -> bool:
        return self.action != "completed"


def process_rss_bytes(pid: int) -> int | None:
    """Return current resident bytes for ``pid``, or ``None`` if it exited.

    Linux uses ``/proc`` without starting another process.  Windows uses
    ``GetProcessMemoryInfo``.  macOS and other POSIX hosts use the standard
    ``ps`` interface because Python's ``resource`` module exposes only a
    high-water value, not a live arbitrary-child RSS sample.
    """

    if isinstance(pid, bool) or int(pid) <= 0:
        raise ValueError("pid must be a positive integer")
    resolved_pid = int(pid)
    if os.name == "nt":
        return _windows_process_rss_bytes(resolved_pid)
    if sys.platform.startswith("linux"):
        statm = Path(f"/proc/{resolved_pid}/statm")
        try:
            fields = statm.read_text(encoding="ascii").split()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ProcessRssError(f"could not read Linux RSS for pid {resolved_pid}") from exc
        if len(fields) < 2:
            raise ProcessRssError(f"Linux RSS record is malformed for pid {resolved_pid}")
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            resident_pages = int(fields[1])
        except (OSError, TypeError, ValueError) as exc:
            raise ProcessRssError(f"Linux RSS record is malformed for pid {resolved_pid}") from exc
        return resident_pages * page_size
    return _posix_ps_rss_bytes(resolved_pid)


def monitor_process(
    process: WatchableProcess,
    *,
    sample_rss: Callable[[int], int | None] = process_rss_bytes,
    sample_callback: Callable[[int, int, int], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    limits: WatchdogLimits | None = None,
) -> WatchdogResult:
    """Monitor one child until it exits or a memory ceiling stops it."""

    active_limits = limits or WatchdogLimits()
    peak_rss = 0
    last_rss = 0
    terminate_requested_at: float | None = None
    action = "completed"

    while True:
        returncode = process.poll()
        if returncode is not None:
            return _result(
                process,
                returncode=returncode,
                action=action,
                peak_rss=peak_rss,
                last_rss=last_rss,
                limits=active_limits,
            )

        sampled = sample_rss(process.pid)
        if sampled is None:
            returncode = process.poll()
            if returncode is not None:
                return _result(
                    process,
                    returncode=returncode,
                    action=action,
                    peak_rss=peak_rss,
                    last_rss=last_rss,
                    limits=active_limits,
                )
            raise ProcessRssError(f"RSS sampler returned no value for live pid {process.pid}")
        if isinstance(sampled, bool):
            raise ProcessRssError(f"RSS sampler returned a non-integer value for pid {process.pid}")
        try:
            last_rss = int(sampled)
        except (TypeError, ValueError) as exc:
            raise ProcessRssError(
                f"RSS sampler returned a non-integer value for pid {process.pid}"
            ) from exc
        if last_rss < 0:
            raise ProcessRssError(f"RSS sampler returned a negative value for pid {process.pid}")
        peak_rss = max(peak_rss, last_rss)
        if sample_callback is not None:
            sample_callback(process.pid, last_rss, peak_rss)
        now = float(monotonic())

        if last_rss >= active_limits.kill_rss_bytes:
            process.kill()
            return _result_after_stop(
                process,
                action="kill_rss",
                peak_rss=peak_rss,
                last_rss=last_rss,
                limits=active_limits,
            )

        if terminate_requested_at is None:
            if last_rss >= active_limits.terminate_rss_bytes:
                process.terminate()
                terminate_requested_at = now
                action = "terminate_rss"
        elif now - terminate_requested_at >= active_limits.terminate_grace_seconds:
            process.kill()
            return _result_after_stop(
                process,
                action="kill_after_grace",
                peak_rss=peak_rss,
                last_rss=last_rss,
                limits=active_limits,
            )

        sleep(active_limits.poll_interval_seconds)


def run_command_with_watchdog(
    command: Sequence[str],
    *,
    output_dir: str | Path,
    popen_factory: Callable[[Sequence[str]], WatchableProcess] = subprocess.Popen,
    sample_rss: Callable[[int], int | None] = process_rss_bytes,
    sample_callback: Callable[[int, int, int], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    limits: WatchdogLimits | None = None,
    completion_callback: Callable[[WatchdogResult], bool] | None = None,
    failure_details_callback: (
        Callable[[WatchdogResult], Mapping[str, object]] | None
    ) = None,
    intervention_details_callback: (
        Callable[[WatchdogResult], Mapping[str, object]] | None
    ) = None,
) -> int:
    """Run a worker command and return its accepted status.

    A successful child may hand promotion back to the parent through
    ``completion_callback``.  The callback receives the peak sampled over the
    child's full lifetime and must return true only after it has either
    promoted an accepted staged run or safely recorded a rejection.
    """

    active_limits = limits or WatchdogLimits()
    process = popen_factory(list(command))
    try:
        result = monitor_process(
            process,
            sample_rss=sample_rss,
            sample_callback=sample_callback,
            monotonic=monotonic,
            sleep=sleep,
            limits=active_limits,
        )
    except Exception as exc:
        stop_action, returncode = _stop_after_monitor_error(process, active_limits)
        result = WatchdogResult(
            child_pid=int(process.pid),
            returncode=returncode,
            action=stop_action,
            peak_rss_bytes=0,
            last_rss_bytes=0,
            acceptance_peak_rss_bytes=active_limits.acceptance_peak_rss_bytes,
            terminate_rss_bytes=active_limits.terminate_rss_bytes,
            kill_rss_bytes=active_limits.kill_rss_bytes,
            poll_interval_seconds=active_limits.poll_interval_seconds,
        )
        intervention_details: Mapping[str, object] | None = None
        cleanup_error: str | None = None
        if intervention_details_callback is not None:
            try:
                intervention_details = intervention_details_callback(result)
            except Exception as cleanup_exc:
                cleanup_error = type(cleanup_exc).__name__
        monitor_error = type(exc).__name__
        if cleanup_error is not None:
            monitor_error = f"{monitor_error};cleanup={cleanup_error}"
        write_watchdog_failure_receipt(
            output_dir,
            result,
            monitor_error=monitor_error,
            worker_failure=intervention_details,
        )
        return 1

    if result.intervened:
        intervention_details = None
        cleanup_error = None
        if intervention_details_callback is not None:
            try:
                intervention_details = intervention_details_callback(result)
            except Exception as exc:
                cleanup_error = f"cleanup={type(exc).__name__}"
        write_watchdog_failure_receipt(
            output_dir,
            result,
            monitor_error=cleanup_error,
            worker_failure=intervention_details,
        )
        return 1
    if result.returncode != 0:
        worker_failure: Mapping[str, object] | None = None
        sidecar_error: str | None = None
        if failure_details_callback is not None:
            try:
                worker_failure = failure_details_callback(result)
            except Exception as exc:
                sidecar_error = type(exc).__name__
        write_watchdog_failure_receipt(
            output_dir,
            replace(result, action="child_exit_nonzero"),
            failure_kind="worker_failure",
            monitor_error=sidecar_error,
            worker_failure=worker_failure,
        )
        return int(result.returncode)
    if completion_callback is not None:
        try:
            accepted = completion_callback(result)
        except Exception as exc:
            intervention_details: Mapping[str, object] | None = None
            cleanup_error: str | None = None
            if intervention_details_callback is not None:
                try:
                    intervention_details = intervention_details_callback(result)
                except Exception as cleanup_exc:
                    cleanup_error = type(cleanup_exc).__name__
            monitor_error = type(exc).__name__
            if cleanup_error is not None:
                monitor_error = (
                    f"{monitor_error};cleanup={cleanup_error}"
                )
            write_watchdog_failure_receipt(
                output_dir,
                replace(result, action="acceptance_callback_error"),
                monitor_error=monitor_error,
                worker_failure=intervention_details,
            )
            return 1
        if accepted is not True:
            if intervention_details_callback is not None:
                intervention_details_callback(result)
            return 1
        if result.peak_rss_bytes > result.acceptance_peak_rss_bytes:
            intervention_details = None
            cleanup_error = None
            if intervention_details_callback is not None:
                try:
                    intervention_details = intervention_details_callback(result)
                except Exception as exc:
                    cleanup_error = type(exc).__name__
            write_watchdog_failure_receipt(
                output_dir,
                replace(result, action="invalid_acceptance_above_peak_limit"),
                monitor_error=cleanup_error,
                worker_failure=intervention_details,
            )
            return 1
        return 0
    if result.peak_rss_bytes > result.acceptance_peak_rss_bytes:
        write_watchdog_failure_receipt(
            output_dir,
            replace(result, action="reject_acceptance_peak"),
        )
        return 1
    return 0


def write_watchdog_failure_receipt(
    output_dir: str | Path,
    result: WatchdogResult,
    *,
    monitor_error: str | None = None,
    failure_kind: str = "parent_memory_watchdog",
    worker_failure: Mapping[str, object] | None = None,
) -> Path:
    """Write a small sibling receipt without promoting a scientific run."""

    requested = Path(output_dir)
    parent = requested.parent
    parent.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(timezone.utc)
    stamp = recorded_at.strftime("%Y%m%dT%H%M%S%fZ")
    name = requested.name or "tdcsim-cbo-run"
    receipt = parent / f"{name}.watchdog-failure-{result.child_pid}-{stamp}.json"
    payload: dict[str, object] = {
        "schema_version": "tdcsim_cbo_watchdog_failure_v1",
        "status": "failed",
        "failure_kind": failure_kind,
        "action": result.action,
        "child_pid": result.child_pid,
        "child_returncode": result.returncode,
        "last_rss_bytes": result.last_rss_bytes,
        "peak_rss_bytes": result.peak_rss_bytes,
        "acceptance_peak_rss_bytes": result.acceptance_peak_rss_bytes,
        "terminate_rss_bytes": result.terminate_rss_bytes,
        "kill_rss_bytes": result.kill_rss_bytes,
        "poll_interval_seconds": result.poll_interval_seconds,
        "recorded_at_utc": recorded_at.isoformat(),
    }
    if monitor_error is not None:
        payload["monitor_error"] = monitor_error
    if worker_failure is not None:
        payload["worker_failure"] = dict(worker_failure)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=parent,
        prefix=f".{name}.watchdog-",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, receipt)
    return receipt


def _result(
    process: WatchableProcess,
    *,
    returncode: int,
    action: str,
    peak_rss: int,
    last_rss: int,
    limits: WatchdogLimits,
) -> WatchdogResult:
    return WatchdogResult(
        child_pid=int(process.pid),
        returncode=int(returncode),
        action=action,
        peak_rss_bytes=peak_rss,
        last_rss_bytes=last_rss,
        acceptance_peak_rss_bytes=limits.acceptance_peak_rss_bytes,
        terminate_rss_bytes=limits.terminate_rss_bytes,
        kill_rss_bytes=limits.kill_rss_bytes,
        poll_interval_seconds=limits.poll_interval_seconds,
    )


def _result_after_stop(
    process: WatchableProcess,
    *,
    action: str,
    peak_rss: int,
    last_rss: int,
    limits: WatchdogLimits,
) -> WatchdogResult:
    try:
        returncode = process.wait(timeout=limits.kill_wait_seconds)
    except subprocess.TimeoutExpired as exc:
        raise ProcessRssError(f"worker pid {process.pid} did not exit after kill") from exc
    return _result(
        process,
        returncode=returncode,
        action=action,
        peak_rss=peak_rss,
        last_rss=last_rss,
        limits=limits,
    )


def _stop_after_monitor_error(
    process: WatchableProcess,
    limits: WatchdogLimits,
) -> tuple[str, int]:
    if process.poll() is not None:
        return "monitor_error_child_exited", int(process.poll() or 0)
    process.terminate()
    try:
        return "monitor_error_terminate", int(
            process.wait(timeout=limits.terminate_grace_seconds)
        )
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            return "monitor_error_kill", int(
                process.wait(timeout=limits.kill_wait_seconds)
            )
        except subprocess.TimeoutExpired as exc:
            raise ProcessRssError(
                f"worker pid {process.pid} did not exit after monitor failure"
            ) from exc


def _posix_ps_rss_bytes(pid: int) -> int | None:
    try:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProcessRssError(f"could not sample POSIX RSS for pid {pid}") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first = completed.stdout.strip().splitlines()[0].strip()
    try:
        rss_kib = int(first)
    except ValueError as exc:
        raise ProcessRssError(f"POSIX RSS record is malformed for pid {pid}") from exc
    return rss_kib * 1024


def _windows_process_rss_bytes(pid: int) -> int | None:
    from ctypes import wintypes

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCountersEx),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(
        process_query_limited_information,
        False,
        pid,
    )
    if not handle:
        error = ctypes.get_last_error()
        if error == 87:  # ERROR_INVALID_PARAMETER: the process no longer exists.
            return None
        raise ProcessRssError(f"could not open Windows process {pid} for RSS sampling")
    try:
        counters = ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            handle,
            ctypes.byref(counters),
            counters.cb,
        ):
            error = ctypes.get_last_error()
            if error == 6:  # ERROR_INVALID_HANDLE: the process exited during sampling.
                return None
            raise ProcessRssError(
                f"could not read Windows RSS for pid {pid}"
            )
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


__all__ = [
    "DEFAULT_ACCEPTANCE_PEAK_RSS_BYTES",
    "DEFAULT_KILL_RSS_BYTES",
    "DEFAULT_TERMINATE_RSS_BYTES",
    "GIB",
    "ProcessRssError",
    "WatchdogLimits",
    "WatchdogResult",
    "monitor_process",
    "process_rss_bytes",
    "run_command_with_watchdog",
    "write_watchdog_failure_receipt",
]
