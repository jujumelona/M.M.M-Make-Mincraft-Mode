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
