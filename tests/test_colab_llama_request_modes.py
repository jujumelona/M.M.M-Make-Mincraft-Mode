from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import colab_mtp_server
from minecraft_mod_ai import llama_server_hardware_policy
from minecraft_mod_ai.colab_llama_request_routing_contract import (
    _transient_managed_failure,
)
from minecraft_mod_ai.model_adapters import ModelBackendError
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def _config(*, model_id: str, filename: str = "model.gguf") -> SimpleNamespace:
    return SimpleNamespace(
        role="planner",
        model_id=model_id,
        max_context=32768,
        max_new_tokens=8192,
        extra={"gguf_filename": filename},
    )


def _request(response_format) -> SimpleNamespace:
    return SimpleNamespace(
        messages=({"role": "user", "content": "hello"},),
        response_format=response_format,
    )


def test_package_bootstrap_installs_request_safe_colab_decode_contract() -> None:
    assert getattr(
        LlamaCppAdapter.generate,
        "_mmm_colab_request_mode_router",
        False,
    ) is True
    assert getattr(
        llama_server_hardware_policy._strict_server_generate,
        "_mmm_local_stream_watchdog",
        False,
    ) is True


@pytest.mark.parametrize(
    "response_format",
    [
        "json",
        "json_object",
        "json_schema",
        {"type": "json_object"},
        {"type": "json_schema"},
    ],
)
def test_structured_output_always_uses_baseline_even_for_mtp_model(
    response_format,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_POLICY", "auto")
    monkeypatch.setattr(colab_mtp_server, "_MTP_DISABLED_REASON", None)
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    assert colab_mtp_server.request_server_mode(config, _request(response_format)) == "baseline"


def test_free_text_can_use_mtp_only_for_mtp_capable_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_POLICY", "auto")
    monkeypatch.setattr(colab_mtp_server, "_MTP_DISABLED_REASON", None)
    mtp = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")
    gemma = _config(model_id="google/gemma-4-12B-it-qat-q4_0-gguf")

    assert colab_mtp_server.request_server_mode(mtp, _request("text")) == "mtp"
    assert colab_mtp_server.request_server_mode(gemma, _request("text")) == "baseline"


def test_runtime_mtp_disable_forces_future_text_requests_to_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MMM_LLAMA_MTP_POLICY", "auto")
    monkeypatch.setattr(colab_mtp_server, "_MTP_DISABLED_REASON", "probe failed")
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    assert colab_mtp_server.request_server_mode(config, _request("text")) == "baseline"


def test_baseline_config_contains_no_speculation_and_applies_selected_kv_type(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "server.json"
    monkeypatch.setattr(colab_mtp_server, "SERVER_CONFIG_PATH", config_path)
    monkeypatch.setattr(colab_mtp_server, "_kv_cache_type_id", lambda: 123)
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    width = colab_mtp_server._write_config(config, "/tmp/model.gguf", mode="baseline")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert width == 0
    model = payload["model"]
    assert "draft_model" not in model
    assert "draft_model_num_pred_tokens" not in model
    assert model["n_ctx"] == colab_mtp_server.SERVER_CONTEXT_CAP
    assert model["type_k"] == 123
    assert model["type_v"] == 123


def test_mtp_config_is_explicit_bounded_and_uses_same_kv_type(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "server.json"
    monkeypatch.setattr(colab_mtp_server, "SERVER_CONFIG_PATH", config_path)
    monkeypatch.setattr(colab_mtp_server, "_kv_cache_type_id", lambda: 456)
    monkeypatch.setenv("MMM_LLAMA_MTP_WIDTH", "3")
    config = _config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")

    width = colab_mtp_server._write_config(config, "/tmp/model.gguf", mode="mtp")
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert width == 3
    model = payload["model"]
    assert model["draft_model"] == "draft-mtp"
    assert model["draft_model_num_pred_tokens"] == 3
    assert model["n_seq_max"] == 1
    assert model["type_k"] == 456
    assert model["type_v"] == 456


def test_invalid_kv_cache_quant_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MMM_KV_CACHE_QUANT", "bogus")
    with pytest.raises(RuntimeError, match="MMM_KV_CACHE_QUANT"):
        colab_mtp_server._kv_cache_quant()


def test_decode_log_summary_distinguishes_first_token_stall_from_server_progress() -> None:
    stalled = colab_mtp_server.decode_log_summary(
        "slot update: n_decoded = 1\nslot update: n_decoded = 1\n"
    )
    progressed = colab_mtp_server.decode_log_summary(
        "slot update: n_decoded = 1\nslot update: n_decoded = 42\n"
    )

    assert "stalled" in stalled
    assert "max=1" in stalled
    assert "advanced beyond the first token" in progressed
    assert "last=42" in progressed


class _StreamResponse:
    status_code = 200
    text = ""

    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self):
        yield from self.lines

    def read(self):
        return b""


def _delta(content: str) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": content}}]}
    )


def test_managed_structured_stream_disables_reasoning_without_server_json_grammar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    payloads: list[dict] = []

    def stream(*args, **kwargs):
        payloads.append(dict(kwargs["json"]))
        return _StreamResponse([_delta('{"game_design":{}}'), "data: [DONE]"])

    monkeypatch.setattr(httpx, "stream", stream)
    adapter = SimpleNamespace(
        config=_config(model_id="unsloth/Qwen3.5-9B-MTP-GGUF")
    )

    result = llama_server_hardware_policy._strict_server_generate(
        adapter,
        _request("json"),
        colab_mtp_server.SERVER_API_URL,
    )

    assert result == '{"game_design":{}}'
    assert payloads
    payload = payloads[0]
    assert "response_format" not in payload
    assert payload["reasoning_effort"] == "none"
    assert payload["stream"] is True


def test_mtp_probe_requires_completed_multi_step_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colab_mtp_server.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse(
            [_delta("1 2 3 "), _delta("4 5 6 7 8 9 10"), "data: [DONE]"]
        ),
    )

    ok, detail = colab_mtp_server._probe_mtp_server()

    assert ok is True
    assert "events=2" in detail


def test_mtp_probe_rejects_first_delta_then_broken_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        colab_mtp_server.httpx,
        "stream",
        lambda *args, **kwargs: _StreamResponse([_delta("1 2 3 4 5")]),
    )

    ok, detail = colab_mtp_server._probe_mtp_server()

    assert ok is False
    assert "before [DONE]" in detail


def test_transient_decode_stall_is_restartable_but_http_400_is_not() -> None:
    stalled = ModelBackendError(
        role="planner",
        model_id="model",
        cause=RuntimeError("llama server produced no output delta for 90s"),
    )
    bad_request = ModelBackendError(
        role="planner",
        model_id="model",
        cause=RuntimeError("llama server returned HTTP 400: malformed request"),
    )

    assert _transient_managed_failure(stalled) is True
    assert _transient_managed_failure(bad_request) is False
