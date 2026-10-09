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
_KERNEL_MONITOR_ENABLED = False
_MEMORY_GUARD_MIB = 2048  # Fail closed before a Colab kernel is OOM-killed.


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
    """Read container memory pressure on either cgroup v2 or legacy v1."""
    events: dict[str, int] = {}
    v2_limit = Path("/sys/fs/cgroup/memory.max")
    if v2_limit.is_file():
        current = _read_int(Path("/sys/fs/cgroup/memory.current"))
        try:
            raw_max = v2_limit.read_text(encoding="utf-8").strip()
            maximum = 0 if raw_max == "max" else max(0, int(raw_max))
        except (OSError, UnicodeError, ValueError):
            maximum = 0
        try:
            for line in Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8").splitlines():
                key, _, raw = line.partition(" ")
                if key and raw:
                    events[key] = int(raw)
        except (OSError, UnicodeError, ValueError):
            pass
        return current, maximum, events

    v1_root = Path("/sys/fs/cgroup/memory")
    current = _read_int(v1_root / "memory.usage_in_bytes")
    maximum = _read_int(v1_root / "memory.limit_in_bytes")
    # v1's huge sentinel encodes "unlimited", not a real container budget.
    if maximum >= 1 << 60:
        maximum = 0
    failcnt = _read_int(v1_root / "memory.failcnt")
    if failcnt:
        events["failcnt"] = failcnt
    return current, maximum, events


def _atomic_write(payload: dict[str, Any]) -> None:
    path = _snapshot_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
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
    # A completely exhausted cgroup has ZERO free bytes. Never treat this
    # as a missing reading and fall back to /proc/meminfo's larger host total.
    effective_available = (
        min(available, cgroup_available) if available
        else cgroup_available
    ) if cgroup_max > 0 else available
    if _MIN_AVAILABLE is None or effective_available < _MIN_AVAILABLE:
        _MIN_AVAILABLE = effective_available
    return {
        "schema_version": "mmm/runtime-memory-snapshot-v1",
        "stage": os.environ.get("MMM_RUNTIME_STAGE", "unknown").strip() or "unknown",
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
            if effective_available < 768 * 1024 * 1024
            else "low"
            if effective_available < 1536 * 1024 * 1024
            else "ok"
        ),
    }


def _memory_reserve_bytes() -> int:
    """Use an explicit Colab reserve; invalid/negative overrides fail closed."""
    raw = os.environ.get("MMM_COLAB_RAM_RESERVE_MIB", str(_MEMORY_GUARD_MIB))
    try:
        mib = int(raw.strip())
    except ValueError as exc:
        raise ValueError("MMM_COLAB_RAM_RESERVE_MIB must be a positive integer") from exc
    if mib < 512:
        raise ValueError("MMM_COLAB_RAM_RESERVE_MIB must be >=512")
    return mib * 1024 * 1024


def assert_memory_headroom(stage: str, *, reserve_bytes: int | None = None) -> dict[str, Any]:
    """Reject new heavyweight allocations when cgroup memory is near exhausted.

    Never kill the notebook kernel. This is a best-effort preflight, not a
    guarantee against allocations occurring between samples.
    """
    sample = _sample(_MANAGED_PID or 0)
    _atomic_write(sample)
    reserve = _memory_reserve_bytes() if reserve_bytes is None else reserve_bytes
    available = int(sample["effective_mem_available_bytes"])
    if available < reserve:
        raise MemoryError(
            f"COLAB_RAM_HEADROOM_EXHAUSTED: stage={stage}; "
            f"available_mib={available // 1048576}; "
            f"reserve_mib={reserve // 1048576}; "
            f"kernel_rss_mib={sample['kernel_rss_bytes'] // 1048576}; "
            f"managed_rss_mib={sample['managed_rss_bytes'] // 1048576}; "
            f"cgroup_oom_kill={sample['cgroup_memory_events'].get('oom_kill', 0)}; "
            f"snapshot={_snapshot_path()}"
        )
    return sample


def _watchdog_loop(stop: threading.Event, interval: float) -> None:
    global _MANAGED_PID
    while not stop.is_set():
        pid = _MANAGED_PID or 0
        sample = _sample(pid)
        _atomic_write(sample)
        # A managed native subprocess is owned by MMM. If the cgroup gets
        # critically low, stop *only* this verified child to protect Jupyter.
        # The notebook kernel and unrelated services are never signaled.
        if (
            pid > 0
            and bool(os.environ.get("MMM_COLAB_SETUP_RECEIPT", "").strip())
            and _process_start_ticks(pid) == sample["managed_start_ticks"]
            and sample["managed_start_ticks"] > 0
            and int(sample["effective_mem_available_bytes"]) < _memory_reserve_bytes()
        ):
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                if b"llama-server" in cmdline:
                    sample["protective_action"] = "terminate_managed_llama_server"
                    _atomic_write(sample)
                    os.kill(pid, signal.SIGTERM)
                    print(
                        "COLAB_RAM_GUARD: terminated managed llama-server before "
                        "critical host memory exhaustion; snapshot="
                        + str(_snapshot_path()),
                        flush=True,
                    )
                    with _LOCK:
                        if _MANAGED_PID == pid:
                            _MANAGED_PID = None
            except (OSError, ProcessLookupError, PermissionError):
                pass
        stop.wait(interval)


def start_kernel_memory_watchdog() -> None:
    """Begin Colab forensics before pip, llama-server, image, and Gradle stages."""
    global _STOP, _THREAD, _KERNEL_MONITOR_ENABLED
    with _LOCK:
        _KERNEL_MONITOR_ENABLED = True
        if _THREAD is not None and _THREAD.is_alive():
            return
        raw = os.environ.get("MMM_RUNTIME_MEMORY_SAMPLE_SECONDS", "1")
        try:
            interval = max(0.25, float(raw))
        except ValueError:
            interval = 1.0
        stop = threading.Event()
        _STOP = stop
        _THREAD = threading.Thread(
            target=_watchdog_loop,
            args=(stop, interval),
            name="mmm_colab_memory_watchdog",
            daemon=True,
        )
        _THREAD.start()


def start_managed_process_watchdog(pid: int) -> None:
    global _MANAGED_PID
    pid = int(pid)
    if pid <= 0:
        return
    with _LOCK:
        _MANAGED_PID = pid
    start_kernel_memory_watchdog()


def stop_managed_process_watchdog() -> None:
    global _MANAGED_PID, _THREAD, _STOP
    with _LOCK:
        _MANAGED_PID = None
        if _KERNEL_MONITOR_ENABLED:
            return
        stop, thread = _STOP, _THREAD
        _STOP, _THREAD = None, None
    if stop:
        stop.set()
    if thread and thread is not threading.current_thread():
        thread.join(timeout=0.5)


def shutdown_kernel_memory_watchdog() -> None:
    """Stop owned monitor before a hot Git checkout replaces this module."""
    global _KERNEL_MONITOR_ENABLED, _MANAGED_PID, _THREAD, _STOP
    with _LOCK:
        _KERNEL_MONITOR_ENABLED = False
        _MANAGED_PID = None
        stop, thread = _STOP, _THREAD
        _STOP, _THREAD = None, None
    if stop is not None:
        stop.set()
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=1.0)


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
        "last_stage": snapshot.get("stage"),
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
    "start_kernel_memory_watchdog",
    "shutdown_kernel_memory_watchdog",
    "assert_memory_headroom",
    "stop_managed_process_watchdog",
]
