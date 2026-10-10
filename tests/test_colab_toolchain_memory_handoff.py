"""Memory handoff from idle Colab planner to Java/Gradle setup."""

from types import SimpleNamespace

from minecraft_mod_ai import llama_server_autotune as autotune
from minecraft_mod_ai import runtime_memory_watchdog as watchdog


def _managed_process():
    return SimpleNamespace(pid=4321, poll=lambda: None)


def test_low_ram_colab_releases_only_owned_idle_server(monkeypatch):
    calls = []
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", '{"ok":true}')
    monkeypatch.delenv("MMM_COLAB_TOOLCHAIN_HEADROOM_MIB", raising=False)
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", _managed_process())
    monkeypatch.setattr(
        watchdog, "_sample",
        lambda _pid: {"effective_mem_available_bytes": 2518 * 1048576},
    )
    monkeypatch.setattr(
        autotune, "_shutdown_managed_server", lambda: calls.append("released")
    )
    assert autotune.release_managed_server_before_toolchain() is True
    assert calls == ["released"]


def test_healthy_ram_keeps_model_resident(monkeypatch):
    calls = []
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", '{"ok":true}')
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", _managed_process())
    monkeypatch.setattr(
        watchdog, "_sample",
        lambda _pid: {"effective_mem_available_bytes": 7000 * 1048576},
    )
    monkeypatch.setattr(
        autotune, "_shutdown_managed_server", lambda: calls.append("released")
    )
    assert autotune.release_managed_server_before_toolchain() is False
    assert not calls


def test_non_colab_or_external_server_is_never_terminated(monkeypatch):
    calls = []
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", _managed_process())
    monkeypatch.setattr(
        autotune, "_shutdown_managed_server", lambda: calls.append("released")
    )
    assert autotune.release_managed_server_before_toolchain() is False
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", '{"ok":true}')
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", None)
    assert autotune.release_managed_server_before_toolchain() is False
    assert calls == []


def test_explicit_headroom_threshold_is_honored(monkeypatch):
    calls = []
    monkeypatch.setenv("MMM_COLAB_SETUP_RECEIPT", '{"ok":true}')
    monkeypatch.setenv("MMM_COLAB_TOOLCHAIN_HEADROOM_MIB", "8192")
    monkeypatch.setattr(autotune, "_MANAGED_PROCESS", _managed_process())
    monkeypatch.setattr(
        watchdog, "_sample",
        lambda _pid: {"effective_mem_available_bytes": 6000 * 1048576},
    )
    monkeypatch.setattr(
        autotune, "_shutdown_managed_server", lambda: calls.append("released")
    )
    assert autotune.release_managed_server_before_toolchain() is True
    assert calls == ["released"]


def test_production_admission_releases_managed_planner_then_resamples(monkeypatch, tmp_path):
    """A loaded native GGUF must be handed off before the hard RAM admission gate."""
    import json

    events = []
    available = [1720]
    monkeypatch.setenv("MMM_COLAB_RAM_RESERVE_MIB", "2048")
    snapshot_path = tmp_path / "production-memory.json"
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(snapshot_path))

    def release_idle_server():
        events.append("release")
        available[0] = 5500
        return True

    def sample(_pid):
        events.append("sample")
        return {
            "effective_mem_available_bytes": available[0] * 1048576,
            "kernel_rss_bytes": 1076 * 1048576,
            "managed_rss_bytes": (9331 if available[0] < 2048 else 0) * 1048576,
            "cgroup_memory_events": {"oom_kill": 0},
        }

    monkeypatch.setattr(autotune, "release_managed_server_before_toolchain", release_idle_server)
    monkeypatch.setattr(watchdog, "_sample", sample)

    result = watchdog.prepare_production_memory_headroom()

    assert events == ["release", "sample"]
    assert result["released_managed_llama_server"] is True
    assert result["preflight_reserve_bytes"] == 2048 * 1048576
    assert result["effective_mem_available_bytes"] == 5500 * 1048576
    persisted = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert persisted["preflight_stage"] == "before_production_build"


def test_production_admission_does_not_bypass_guard_if_no_owned_model(monkeypatch, tmp_path):
    import pytest

    monkeypatch.setenv("MMM_COLAB_RAM_RESERVE_MIB", "2048")
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(tmp_path / "memory.json"))
    calls = []
    monkeypatch.setattr(
        autotune, "release_managed_server_before_toolchain",
        lambda: calls.append("release_attempted") or False,
    )
    monkeypatch.setattr(
        watchdog, "_sample",
        lambda _pid: {
            "effective_mem_available_bytes": 1720 * 1048576,
            "kernel_rss_bytes": 1076 * 1048576,
            "managed_rss_bytes": 0,
            "cgroup_memory_events": {"oom_kill": 0},
        },
    )

    with pytest.raises(MemoryError, match="available_mib=1720"):
        watchdog.prepare_production_memory_headroom()
    assert calls == ["release_attempted"]


def test_production_admission_keeps_healthy_managed_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_COLAB_RAM_RESERVE_MIB", "2048")
    monkeypatch.setenv("MMM_RUNTIME_MEMORY_SNAPSHOT", str(tmp_path / "memory.json"))
    monkeypatch.setattr(autotune, "release_managed_server_before_toolchain", lambda: False)
    monkeypatch.setattr(
        watchdog, "_sample",
        lambda _pid: {
            "effective_mem_available_bytes": 7000 * 1048576,
            "kernel_rss_bytes": 900 * 1048576,
            "managed_rss_bytes": 9331 * 1048576,
            "cgroup_memory_events": {"oom_kill": 0},
        },
    )

    result = watchdog.prepare_production_memory_headroom()
    assert result["released_managed_llama_server"] is False
    assert result["effective_mem_available_bytes"] == 7000 * 1048576
