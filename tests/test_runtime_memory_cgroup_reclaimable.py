"""cgroup working-set and managed-server shutdown regression coverage."""

import signal
from pathlib import Path

import pytest

from minecraft_mod_ai import llama_server_runtime_tuning as tuning
from minecraft_mod_ai import runtime_memory_watchdog as watchdog


@pytest.mark.parametrize(
    ("current", "limit", "inactive_file", "expected"),
    [
        (9_000, 10_000, 6_000, 7_000),
        (9_000, 10_000, 0, 1_000),
        (12_000, 10_000, 1_000, 0),
        (1_000, 10_000, 20_000, 10_000),
        (10_000, 0, 10_000, 0),
    ],
)
def test_cgroup_headroom_only_discounts_inactive_file(
    current, limit, inactive_file, expected
):
    assert watchdog._cgroup_effective_headroom_bytes(
        current, limit, inactive_file
    ) == expected


def test_memory_stat_reads_only_named_reclaimable_counter(tmp_path):
    stat = tmp_path / "memory.stat"
    stat.write_text("file 8000\ninactive_file 2000\nactive_file 6000\n")
    assert watchdog._read_memory_stat(stat, "inactive_file") == 2000
    assert watchdog._read_memory_stat(stat, "not_a_stat") == 0
    assert watchdog._read_memory_stat(tmp_path / "absent", "inactive_file") == 0


def test_watchdog_sample_includes_reclaimable_cache_in_effective_headroom(monkeypatch):
    monkeypatch.setattr(watchdog, "_meminfo", lambda: (10_000, 8_000))
    monkeypatch.setattr(watchdog, "_cgroup_memory", lambda: (9_000, 10_000, {}))
    monkeypatch.setattr(
        watchdog, "_reclaimable_cgroup_file_cache_bytes", lambda: 6_000
    )
    monkeypatch.setattr(watchdog, "_process_start_ticks", lambda _pid: 1)
    monkeypatch.setattr(watchdog, "_process_rss_bytes", lambda _pid: 0)
    monkeypatch.setattr(watchdog, "_MIN_AVAILABLE", None)
    sample = watchdog._sample(1234)
    assert sample["cgroup_reclaimable_inactive_file_bytes"] == 6_000
    assert sample["cgroup_effective_headroom_bytes"] == 7_000
    assert sample["effective_mem_available_bytes"] == 7_000


def test_server_sizing_uses_watchdogs_same_memory_authority(monkeypatch):
    monkeypatch.setattr(
        watchdog, "cgroup_effective_headroom_bytes", lambda: 8_192
    )
    assert tuning._cgroup_memory_available_bytes() == 8_192


def test_managed_shutdown_rejects_pid_reuse_without_signaling(monkeypatch):
    signaled = []
    monkeypatch.setattr(watchdog, "_process_start_ticks", lambda _pid: 987)
    monkeypatch.setattr(watchdog.os, "kill", lambda pid, sig: signaled.append((pid, sig)))
    assert watchdog._terminate_verified_managed_llama(1234, 123) == (
        "not_owned_or_already_exited"
    )
    assert signaled == []


def test_managed_shutdown_escalates_stuck_child_with_bounded_grace(monkeypatch):
    signaled = []
    monkeypatch.setattr(watchdog, "_process_start_ticks", lambda _pid: 123)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: b"llama-server --model test.gguf")
    monkeypatch.setattr(watchdog.os, "kill", lambda pid, sig: signaled.append((pid, sig)))
    outcome = watchdog._terminate_verified_managed_llama(
        1234, 123, grace_seconds=0
    )
    assert outcome == "sigkill_sent"
    assert signaled == [(1234, signal.SIGTERM), (1234, signal.SIGKILL)]


def test_managed_shutdown_stops_after_sigterm_when_child_exits(monkeypatch):
    signaled = []
    alive = {"ticks": 123}
    monkeypatch.setattr(watchdog, "_process_start_ticks", lambda _pid: alive["ticks"])
    monkeypatch.setattr(Path, "read_bytes", lambda _self: b"llama-server")
    def fake_kill(pid, sig):
        signaled.append((pid, sig))
        if sig == signal.SIGTERM:
            alive["ticks"] = 0
    monkeypatch.setattr(watchdog.os, "kill", fake_kill)
    assert watchdog._terminate_verified_managed_llama(
        1234, 123, grace_seconds=0.1
    ) == "sigterm_exited"
    assert signaled == [(1234, signal.SIGTERM)]
