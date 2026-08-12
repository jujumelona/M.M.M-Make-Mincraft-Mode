from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import colab_mtp_server
from minecraft_mod_ai import colab_server_config_contract


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        max_context=32768,
        max_new_tokens=8192,
        extra={"gguf_filename": "model.gguf"},
    )


def test_package_bootstrap_binds_managed_server_reuse_to_decode_config() -> None:
    assert getattr(
        colab_mtp_server.start_colab_mtp_server,
        "_mmm_server_config_bound",
        False,
    ) is True


def test_kv_cache_change_restarts_same_mode_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"running": False, "mode": None, "stops": 0, "starts": 0}

    def current_start(config, *, mode="baseline"):
        state["running"] = True
        state["mode"] = mode
        state["starts"] += 1
        return "http://127.0.0.1:8910/v1"

    def stop(*, keep_enabled=True):
        state["running"] = False
        state["mode"] = None
        state["stops"] += 1

    fake = SimpleNamespace(
        SERVER_CONTEXT_CAP=16384,
        start_colab_mtp_server=current_start,
        stop_colab_mtp_server=stop,
        colab_mtp_server_running=lambda: bool(state["running"]),
        current_server_mode=lambda: state["mode"],
        _mtp_capable=lambda config: True,
        _mtp_width=lambda: 3,
        _kv_cache_quant=lambda: os.environ.get("MMM_KV_CACHE_QUANT", "q4_0"),
    )

    monkeypatch.setattr(colab_server_config_contract, "_SERVER_KEY", None)
    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q4_0")
    colab_server_config_contract.install(fake)
    fake.start_colab_mtp_server(_config(), mode="baseline")

    assert state["starts"] == 1
    assert state["stops"] == 0

    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "q8_0")
    fake.start_colab_mtp_server(_config(), mode="baseline")

    assert state["starts"] == 2
    assert state["stops"] == 1
    assert state["mode"] == "baseline"
