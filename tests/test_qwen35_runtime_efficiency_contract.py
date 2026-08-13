from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import qwen35_runtime_efficiency_contract as contract


def _config():
    return SimpleNamespace(
        model_id="unsloth/Qwen3.5-9B-MTP-GGUF",
        extra={"gguf_filename": "Qwen3.5-9B-UD-Q4_K_XL.gguf"},
    )


def test_qwen_output_is_unbounded_by_default_and_operator_can_recap(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", raising=False)
    hardware = SimpleNamespace(
        _server_payload=lambda adapter, request: {"max_tokens": 8192}
    )
    contract._install_output_policy(hardware)
    adapter = SimpleNamespace(config=_config())

    assert hardware._server_payload(adapter, object())["max_tokens"] == -1

    monkeypatch.setenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", "24576")
    assert hardware._server_payload(adapter, object())["max_tokens"] == 24576


def test_output_limit_rejects_zero_and_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", "0")
    with pytest.raises(ValueError):
        contract._output_token_limit()
    monkeypatch.setenv("MMM_QWEN35_MAX_OUTPUT_TOKENS", "bad")
    with pytest.raises(ValueError):
        contract._output_token_limit()


def test_fast_tuning_defaults_remove_duplicate_reload_stages(monkeypatch) -> None:
    monkeypatch.delenv("MMM_QWEN35_MTP_TUNING", raising=False)
    defaults = contract._fast_tuning_defaults()

    assert defaults["MMM_LLAMA_MTP_WIDTHS"] == "2,4,8"
    assert defaults["MMM_LLAMA_MTP_CONFIDENCE_WIDTHS"] == ""
    assert defaults["MMM_LLAMA_MTP_P_MIN_CANDIDATES"] == "0"
    assert defaults["MMM_LLAMA_UBATCH_CANDIDATES"] == "512"
    assert defaults["MMM_LLAMA_NGRAM_SPEC_TYPES"] == ""
    assert defaults["MMM_QWEN35_MTP_DRAFT_KV"] == "q4_0"
    assert defaults["MMM_LLAMA_AUTOTUNE_TOKENS"] == "96"


def test_exhaustive_tuning_restores_generic_search(monkeypatch) -> None:
    monkeypatch.setenv("MMM_QWEN35_MTP_TUNING", "exhaustive")
    assert contract._fast_tuning_defaults() == {}


def test_fast_cold_policy_is_temporary_and_respects_explicit_values(monkeypatch) -> None:
    for name in contract._fast_tuning_defaults():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTHS", "1,8")
    seen: dict[str, str | None] = {}

    def ensure(config, request):
        del config, request
        seen["widths"] = os.environ.get("MMM_LLAMA_MTP_WIDTHS")
        seen["p_min"] = os.environ.get("MMM_LLAMA_MTP_P_MIN_CANDIDATES")
        seen["draft_kv"] = os.environ.get("MMM_QWEN35_MTP_DRAFT_KV")
        return "http://127.0.0.1:8910/v1"

    autotune = SimpleNamespace(
        ensure_tuned_server=ensure,
        _MANAGED_PROCESS=None,
        _MANAGED_URL=None,
    )
    contract._install_cold_tuning_policy(autotune)

    assert autotune.ensure_tuned_server(_config(), object()).endswith("/v1")
    assert seen == {"widths": "1,8", "p_min": "0", "draft_kv": "q4_0"}
    assert os.environ["MMM_LLAMA_MTP_WIDTHS"] == "1,8"
    assert "MMM_LLAMA_MTP_P_MIN_CANDIDATES" not in os.environ
    assert "MMM_QWEN35_MTP_DRAFT_KV" not in os.environ
