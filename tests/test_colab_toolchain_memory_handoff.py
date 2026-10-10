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
