from __future__ import annotations

import os
from types import SimpleNamespace

from minecraft_mod_ai.qwen35_mtp_hotpath_contract import install


def _config():
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
        max_context=262144,
    )


def test_cold_tuner_sees_effective_context_and_wide_mtp_candidates(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.delenv("MMM_QWEN35_MTP_CTX", raising=False)
    monkeypatch.delenv("MMM_LLAMA_MTP_WIDTHS", raising=False)
    monkeypatch.setenv("MMM_LLAMA_SERVER_CTX", "16384")
    seen = {}

    def measured(config, request):
        seen["ctx"] = os.environ.get("MMM_LLAMA_SERVER_CTX")
        seen["widths"] = os.environ.get("MMM_LLAMA_MTP_WIDTHS")
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        ensure_tuned_server=measured,
        _base_args=lambda *args: [],
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
    )
    install(autotune)
    assert autotune.ensure_tuned_server(_config(), object()).endswith("/v1")
    assert seen == {"ctx": "0", "widths": "1,2,3,4,5,6,8"}
    assert os.environ["MMM_LLAMA_SERVER_CTX"] == "16384"
    assert "MMM_LLAMA_MTP_WIDTHS" not in os.environ


def test_explicit_qwen_context_and_widths_are_respected(monkeypatch) -> None:
    monkeypatch.delenv("LLAMA_SERVER_URL", raising=False)
    monkeypatch.delenv("MMM_COLAB_SETUP_RECEIPT", raising=False)
    monkeypatch.setenv("MMM_QWEN35_MTP_CTX", "24576")
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTHS", "2,4,8")
    seen = {}

    def measured(config, request):
        seen["ctx"] = os.environ.get("MMM_LLAMA_SERVER_CTX")
        seen["widths"] = os.environ.get("MMM_LLAMA_MTP_WIDTHS")
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        ensure_tuned_server=measured,
        _base_args=lambda *args: [],
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
    )
    install(autotune)
    autotune.ensure_tuned_server(_config(), object())
    assert seen == {"ctx": "24576", "widths": "2,4,8"}
    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "2,4,8"
