from __future__ import annotations

import json
import os

from minecraft_mod_ai import runtime_memory_watchdog as watchdog


def test_previous_kernel_snapshot_reports_cgroup_oom_increment(monkeypatch, tmp_path):
    path = tmp_path / "runtime-memory-last.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "mmm/runtime-memory-snapshot-v1",
                "kernel_pid": 999999,
                "kernel_start_ticks": 1,
                "managed_pid": 1234,
                "effective_mem_available_bytes": 256 * 1024 * 1024,
                "minimum_effective_mem_available_bytes": 128 * 1024 * 1024,
                "kernel_rss_bytes": 3 * 1024 * 1024 * 1024,
                "managed_rss_bytes": 6 * 1024 * 1024 * 1024,
                "cgroup_memory_events": {"oom": 4, "oom_kill": 2},
                "pressure": "critical",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(path))
    monkeypatch.setattr(
        watchdog,
        "_cgroup_memory",
        lambda: (0, 0, {"oom": 5, "oom_kill": 3}),
    )

    diagnostic = watchdog.previous_kernel_crash_diagnostic()

    assert diagnostic["cgroup_oom_kill_increased"] is True
    assert diagnostic["previous_cgroup_oom_kill"] == 2
    assert diagnostic["current_cgroup_oom_kill"] == 3
    assert diagnostic["last_pressure"] == "critical"
    assert diagnostic["last_effective_mem_available_bytes"] == 256 * 1024 * 1024


def test_atomic_snapshot_persists_last_memory_state(monkeypatch, tmp_path):
    path = tmp_path / "last.json"
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(path))
    payload = {
        "schema_version": "mmm/runtime-memory-snapshot-v1",
        "kernel_pid": os.getpid(),
        "effective_mem_available_bytes": 123,
    }

    watchdog._atomic_write(payload)

    assert watchdog.read_last_snapshot() == payload


def test_cleanup_ignores_snapshot_for_current_kernel(monkeypatch, tmp_path):
    path = tmp_path / "last.json"
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(path))
    start = watchdog._process_start_ticks(os.getpid())
    path.write_text(
        json.dumps(
            {
                "kernel_pid": os.getpid(),
                "kernel_start_ticks": start,
                "managed_pid": os.getpid(),
                "managed_start_ticks": start,
            }
        ),
        encoding="utf-8",
    )

    assert watchdog.cleanup_orphaned_managed_process() == {}


def test_cgroup_zero_available_is_critical_not_host_free_memory(monkeypatch):
    monkeypatch.setattr(watchdog, "_meminfo", lambda: (16 * 1024**3, 10 * 1024**3))
    monkeypatch.setattr(
        watchdog, "_cgroup_memory",
        lambda: (12 * 1024**3, 12 * 1024**3, {"oom": 2, "oom_kill": 0}),
    )
    snapshot = watchdog._sample(0)
    assert snapshot["effective_mem_available_bytes"] == 0
    assert snapshot["pressure"] == "critical"


def test_preflight_fails_before_production_with_memory_evidence(monkeypatch, tmp_path):
    import pytest

    path = tmp_path / "ram-last.json"
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(path))
    monkeypatch.setenv("MMM_COLAB_RAM_RESERVE_MIB", "2048")
    monkeypatch.setattr(watchdog, "_meminfo", lambda: (16 * 1024**3, 9 * 1024**3))
    monkeypatch.setattr(
        watchdog, "_cgroup_memory",
        lambda: (11 * 1024**3, 12 * 1024**3, {"oom": 3, "oom_kill": 1}),
    )
    with pytest.raises(MemoryError, match="COLAB_RAM_HEADROOM_EXHAUSTED") as err:
        watchdog.assert_memory_headroom("before_production_build")
    assert "reserve_mib=2048" in str(err.value)
    assert "available_mib=1024" in str(err.value)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["effective_mem_available_bytes"] == 1024**3
    assert saved["cgroup_memory_events"]["oom_kill"] == 1


def test_guard_does_not_kill_unowned_process_or_without_colab(monkeypatch, tmp_path):
    import threading

    sample = {
        "effective_mem_available_bytes": 256 * 1024**2,
        "managed_start_ticks": 100,
    }
    killed = []
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.setattr(watchdog, "_sample", lambda pid: sample)
    monkeypatch.setattr(watchdog, "_atomic_write", lambda snapshot: None)
    monkeypatch.setattr(watchdog, "_process_start_ticks", lambda pid: 100)
    monkeypatch.setattr(watchdog.os, "kill", lambda pid, signal: killed.append(pid))
    stop = threading.Event()
    monkeypatch.setattr(stop, "wait", lambda interval: stop.set())
    monkeypatch.setattr(watchdog, "_MANAGED_PID", 12345)
    watchdog._watchdog_loop(stop, 0.1)
    assert killed == []
