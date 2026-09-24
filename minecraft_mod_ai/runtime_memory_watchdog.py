from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()
_STOP: threading.Event | None = None
_THREAD: threading.Thread | None = None
_MANAGED_PID: int | None = None
_MIN_AVAILABLE: int | None = None


def _snapshot_path() -> Path:
    explicit = os.environ.get("MMM_RUNTIME_MEMORY_SNAPSHOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (Path.cwd() / ".mmm" / "traces" / "runtime-memory-last.json").resolve()


def _read_int(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeError, ValueError):
        return 0


def _process_start_ticks(pid: int) -> int:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return int(fields[21]) if len(fields) > 21 else 0
    except (OSError, UnicodeError, ValueError):
        return 0


def _process_rss_bytes(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return max(0, int(line.split()[1])) * 1024
    except (OSError, UnicodeError, IndexError, ValueError):
        pass
    return 0


def _meminfo() -> tuple[int, int]:
    total = 0
    available = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                total = max(0, int(line.split()[1])) * 1024
            elif line.startswith("MemAvailable:"):
                available = max(0, int(line.split()[1])) * 1024
    except (OSError, UnicodeError, IndexError, ValueError):
        pass
    return total, available


def _cgroup_memory() -> tuple[int, int, dict[str, int]]:
    current = _read_int(Path("/sys/fs/cgroup/memory.current"))
    max_path = Path("/sys/fs/cgroup/memory.max")
    try:
        raw_max = max_path.read_text(encoding="utf-8").strip()
        maximum = 0 if raw_max == "max" else max(0, int(raw_max))
    except (OSError, UnicodeError, ValueError):
        maximum = 0
    events: dict[str, int] = {}
    try:
        for line in Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8").splitlines():
            key, _, raw = line.partition(" ")
            if key and raw:
                events[key] = int(raw)
    except (OSError, UnicodeError, ValueError):
        pass
    return current, maximum, events


def _atomic_write(payload: dict[str, Any]) -> None:
    path = _snapshot_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except OSError:
        return


def _sample(managed_pid: int) -> dict[str, Any]:
    global _MIN_AVAILABLE

    total, available = _meminfo()
    cgroup_current, cgroup_max, cgroup_events = _cgroup_memory()
    cgroup_available = (
        max(0, cgroup_max - cgroup_current)
        if cgroup_max > 0
        else 0
    )
    effective_available = available
    if cgroup_available:
        effective_available = (
            min(effective_available, cgroup_available)
            if effective_available
            else cgroup_available
        )
    if _MIN_AVAILABLE is None or (
        effective_available and effective_available < _MIN_AVAILABLE
    ):
        _MIN_AVAILABLE = effective_available
    return {
        "schema_version": "mmm/runtime-memory-snapshot-v1",
        "sampled_at_unix": time.time(),
        "kernel_pid": os.getpid(),
        "kernel_start_ticks": _process_start_ticks(os.getpid()),
        "managed_pid": managed_pid,
        "managed_start_ticks": _process_start_ticks(managed_pid),
        "kernel_rss_bytes": _process_rss_bytes(os.getpid()),
        "managed_rss_bytes": _process_rss_bytes(managed_pid),
        "system_mem_total_bytes": total,
        "system_mem_available_bytes": available,
        "cgroup_memory_current_bytes": cgroup_current,
        "cgroup_memory_max_bytes": cgroup_max,
        "effective_mem_available_bytes": effective_available,
        "minimum_effective_mem_available_bytes": _MIN_AVAILABLE or 0,
        "cgroup_memory_events": cgroup_events,
        "pressure": (
            "critical"
            if effective_available and effective_available < 768 * 1024 * 1024
            else "low"
            if effective_available and effective_available < 1536 * 1024 * 1024
            else "ok"
        ),
    }


def start_managed_process_watchdog(pid: int) -> None:
    global _STOP, _THREAD, _MANAGED_PID, _MIN_AVAILABLE

    pid = int(pid)
    if pid <= 0:
        return
    stop_managed_process_watchdog()
    interval_raw = os.environ.get("MMM_RUNTIME_MEMORY_SAMPLE_SECONDS", "2").strip()
    try:
        interval = max(0.5, float(interval_raw))
    except ValueError:
        interval = 2.0
    stop = threading.Event()
    with _LOCK:
        _STOP = stop
        _MANAGED_PID = pid
        _MIN_AVAILABLE = None

        def run() -> None:
            while not stop.is_set():
                _atomic_write(_sample(pid))
                if not Path(f"/proc/{pid}").exists():
                    return
                stop.wait(interval)

        thread = threading.Thread(
            target=run,
            name="mmm_runtime_memory_watchdog",
            daemon=True,
        )
        _THREAD = thread
        thread.start()


def stop_managed_process_watchdog() -> None:
    global _STOP, _THREAD, _MANAGED_PID

    with _LOCK:
        stop = _STOP
        thread = _THREAD
        _STOP = None
        _THREAD = None
        _MANAGED_PID = None
    if stop is not None:
        stop.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=0.5)


def read_last_snapshot() -> dict[str, Any]:
    try:
        value = json.loads(_snapshot_path().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def previous_kernel_crash_diagnostic() -> dict[str, Any]:
    snapshot = read_last_snapshot()
    if not snapshot:
        return {}
    old_pid = int(snapshot.get("kernel_pid") or 0)
    old_start = int(snapshot.get("kernel_start_ticks") or 0)
    if (
        old_pid == os.getpid()
        and old_start
        and old_start == _process_start_ticks(os.getpid())
    ):
        return {}
    _current, _maximum, current_events = _cgroup_memory()
    previous_events = snapshot.get("cgroup_memory_events")
    previous_events = previous_events if isinstance(previous_events, dict) else {}
    prior_oom_kill = int(previous_events.get("oom_kill") or 0)
    current_oom_kill = int(current_events.get("oom_kill") or 0)
    return {
        "previous_kernel_pid": old_pid or None,
        "previous_managed_pid": int(snapshot.get("managed_pid") or 0) or None,
        "last_effective_mem_available_bytes": int(
            snapshot.get("effective_mem_available_bytes") or 0
        ),
        "minimum_effective_mem_available_bytes": int(
            snapshot.get("minimum_effective_mem_available_bytes") or 0
        ),
        "last_kernel_rss_bytes": int(snapshot.get("kernel_rss_bytes") or 0),
        "last_managed_rss_bytes": int(snapshot.get("managed_rss_bytes") or 0),
        "previous_cgroup_oom_kill": prior_oom_kill,
        "current_cgroup_oom_kill": current_oom_kill,
        "cgroup_oom_kill_increased": current_oom_kill > prior_oom_kill,
        "last_pressure": snapshot.get("pressure"),
    }


def cleanup_orphaned_managed_process() -> dict[str, Any]:
    snapshot = read_last_snapshot()
    if not snapshot:
        return {}
    old_kernel_pid = int(snapshot.get("kernel_pid") or 0)
    old_kernel_start = int(snapshot.get("kernel_start_ticks") or 0)
    if (
        old_kernel_pid == os.getpid()
        and old_kernel_start
        and old_kernel_start == _process_start_ticks(os.getpid())
    ):
        return {}

    pid = int(snapshot.get("managed_pid") or 0)
    start_ticks = int(snapshot.get("managed_start_ticks") or 0)
    if pid <= 0 or start_ticks <= 0 or not Path(f"/proc/{pid}").exists():
        return {}
    if _process_start_ticks(pid) != start_ticks:
        return {}
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
            "utf-8",
            errors="replace",
        )
    except OSError:
        return {}
    if "llama-server" not in cmdline:
        return {}

    outcome = "terminated"
    try:
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 2.0
        while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        if Path(f"/proc/{pid}").exists():
            os.kill(pid, signal.SIGKILL)
            outcome = "killed"
    except ProcessLookupError:
        outcome = "already_exited"
    except PermissionError:
        outcome = "permission_denied"
    return {
        "pid": pid,
        "outcome": outcome,
        "previous_kernel_pid": old_kernel_pid or None,
        "snapshot_path": str(_snapshot_path()),
    }


__all__ = [
    "cleanup_orphaned_managed_process",
    "previous_kernel_crash_diagnostic",
    "read_last_snapshot",
    "start_managed_process_watchdog",
    "stop_managed_process_watchdog",
]
