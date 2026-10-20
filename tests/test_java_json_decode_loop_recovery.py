"""Real failure from production (17): periodic source JSON cannot use text-only replay."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import llama_server_hardware_policy as hardware
from minecraft_mod_ai.model_adapters import llama_cpp_adapter as adapter_module
from minecraft_mod_ai.model_adapters.base import (
    AdapterConfig, GenerationRequest, GenerationResponse, ModelBackendError,
)
from minecraft_mod_ai.llama_completion_liveness_contract import LlamaDegenerateOutputError


def _request(*, source=True, schema=None):
    return GenerationRequest(
        messages=({"role": "user", "content": "Implement one framed Java member"},),
        response_format="json", response_schema=schema,
        metadata=(
            {"mmm_source_implementation": True, "mmm_strict_source_data": True}
            if source else {"mmm_strict_structured_data": True}
        ),
    )


def test_host_framed_java_json_loop_is_replayed_once_with_new_decode_profile(monkeypatch):
    adapter = adapter_module.LlamaCppAdapter(
        AdapterConfig(role="coder", adapter="llama_cpp", model_id="test"),
    )
    monkeypatch.setattr(adapter, "_server_url", lambda _request: "http://127.0.0.1:80/v1")
    seen = []

    def turn(_adapter, url, request):
        seen.append(request)
        if len(seen) == 1:
            raise LlamaDegenerateOutputError("LLAMA_DEGENERATE_REPEATED_OUTPUT")
        return GenerationResponse(content='{"implementation":"return 7;"}')

    monkeypatch.setattr(adapter_module, "_generate_one_turn", turn)
    value = adapter.generate_turn(_request())
    assert value.content == '{"implementation":"return 7;"}'
    assert len(seen) == 2
    assert seen[1].metadata["mmm_decode_repetition_recovery"] is True
    assert seen[1].metadata["mmm_strict_source_data"] is True
    assert seen[1].response_format == "json"
    assert seen[1].response_schema is None


def test_structured_planner_json_is_not_replayed_as_free_java(monkeypatch):
    adapter = adapter_module.LlamaCppAdapter(
        AdapterConfig(role="coder", adapter="llama_cpp", model_id="test"),
    )
    monkeypatch.setattr(adapter, "_server_url", lambda _request: "http://127.0.0.1:80/v1")
    seen = []

    def turn(_adapter, url, request):
        seen.append(request)
        raise LlamaDegenerateOutputError("LLAMA_DEGENERATE_REPEATED_OUTPUT")

    monkeypatch.setattr(adapter_module, "_generate_one_turn", turn)
    with pytest.raises(ModelBackendError):
        adapter.generate_turn(_request(
            source=False, schema={"type": "object", "properties": {}, "required": []},
        ))
    assert len(seen) == 1


def test_native_payload_changes_sampling_only_for_framed_source_json():
    adapter = SimpleNamespace(config=SimpleNamespace(
        role="coder", max_new_tokens=4096, max_context=32768,
        model_id="bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF",
        extra={"runtime_contract": "mimo"},
    ))
    src = _request()
    src = GenerationRequest(
        messages=src.messages, response_format=src.response_format,
        metadata={**src.metadata, "mmm_decode_repetition_recovery": True},
    )
    replay = hardware._server_payload(adapter, src)
    assert replay["temperature"] == 0.35
    assert replay["repeat_penalty"] == 1.20
    assert replay["response_format"]["type"] == "json_object"
    strict = GenerationRequest(
        messages=src.messages, response_format="json",
        response_schema={"type": "object", "properties": {}},
        metadata={"mmm_decode_repetition_recovery": True},
    )
    normal = hardware._server_payload(adapter, strict)
    assert normal.get("repeat_penalty") != 1.20
