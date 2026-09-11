from __future__ import annotations

import os
from types import SimpleNamespace

import minecraft_mod_ai.qwen35_mtp_hotpath_contract as hotpath


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        extra={
            "runtime_contract": "qwen",
            "decode_hotpath": "t4_mtp",
        }
    )


def _autotune(observed: dict[str, str | None]) -> SimpleNamespace:
    def current(config: object, request: object) -> str:
        del config, request
        observed["server_ctx"] = os.environ.get("MMM_LLAMA_SERVER_CTX")
        observed["active_tuning"] = os.environ.get("MMM_QWEN35_MTP_ACTIVE_TUNING")
        return "ok"

    return SimpleNamespace(
        ensure_tuned_server=current,
        _MANAGED_PROCESS=None,
        _MANAGED_URL="",
    )


def _isolate_runtime(monkeypatch: object) -> None:
    monkeypatch.setattr(hotpath, "_disable_decode_slot_polling", lambda: None)
    monkeypatch.setattr(hotpath, "_reclaim_prior_mmm_server", lambda: None)
    monkeypatch.delenv("MMM_LLAMA_SERVER_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_MTP_WIDTHS", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_ACTIVE_TUNING", raising=False)


def test_unset_context_override_preserves_native_context(monkeypatch: object) -> None:
    _isolate_runtime(monkeypatch)
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    observed: dict[str, str | None] = {}
    autotune = _autotune(observed)

    assert hotpath._context_size(_config()) is None
    hotpath.install(autotune)

    assert autotune.ensure_tuned_server(_config(), object()) == "ok"
    assert observed == {"server_ctx": None, "active_tuning": "1"}
    assert "MMM_LLAMA_SERVER_CTX" not in os.environ
    assert "MMM_QWEN35_MTP_ACTIVE_TUNING" not in os.environ


def test_positive_context_override_is_applied_only_for_call(monkeypatch: object) -> None:
    _isolate_runtime(monkeypatch)
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "4096")
    observed: dict[str, str | None] = {}
    autotune = _autotune(observed)

    assert hotpath._context_size(_config()) == 4096
    hotpath.install(autotune)

    assert autotune.ensure_tuned_server(_config(), object()) == "ok"
    assert observed == {"server_ctx": "4096", "active_tuning": "1"}
    assert "MMM_LLAMA_SERVER_CTX" not in os.environ
    assert "MMM_QWEN35_MTP_ACTIVE_TUNING" not in os.environ
