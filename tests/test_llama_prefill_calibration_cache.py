from __future__ import annotations

import pytest

from minecraft_mod_ai.llama_finish_reason_contract import (
    OUTPUT_EXHAUSTED,
    LlamaCompletionBoundaryError,
)
from minecraft_mod_ai.model_adapters import llama_cpp_adapter as adapter_module
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest, ModelBackendError
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import LlamaCppAdapter


def _adapter() -> LlamaCppAdapter:
    return LlamaCppAdapter(
        AdapterConfig(
            role="coder",
            adapter="llama_cpp",
            model_id="test/model",
            max_new_tokens=8192,
            extra={
                "runtime_contract": "qwen",
                "qwen_family": "qwen3.5",
                "qwen_tool_markup": "qwen3_coder_xml",
                "qwen_action_thinking_control": "enable_thinking_false",
                "qwen_preserve_thinking": False,
                "qwen_reasoning_effort": False,
                "qwen_assistant_prefill": True,
            },
        )
    )


def test_retired_prefill_regeneration_surface_is_absent() -> None:
    for name in (
        "_completion_message_with_prefill",
        "_calibrate_assistant_prefill_generation_prompt",
        "_cached_assistant_prefill_generation_prompt",
        "_assistant_prefill_server_identity",
    ):
        assert not hasattr(adapter_module, name)
    assert not hasattr(LlamaCppAdapter, "_prefill_template_prefix_cache")


def test_output_exhaustion_is_not_semantically_regenerated(monkeypatch) -> None:
    calls = 0

    monkeypatch.setattr(LlamaCppAdapter, "_server_url", lambda self, request: "http://unit.test/v1")
    from minecraft_mod_ai import llama_exact_context
    from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract

    monkeypatch.setattr(
        llama_exact_context,
        "capacity_safe_payload",
        lambda _server_url, payload, **_kwargs: dict(payload),
    )
    monkeypatch.setattr(stream_contract, "_report_server_connection", lambda _url: None)

    def fail_once(_url, _payload):
        nonlocal calls
        calls += 1
        raise LlamaCompletionBoundaryError(
            "typed boundary",
            kind=OUTPUT_EXHAUSTED,
            partial_message={"role": "assistant", "content": "partial"},
            prompt_tokens=100,
            completion_tokens=8192,
            max_tokens=8192,
        )

    monkeypatch.setattr(adapter_module, "_completion_message", fail_once)

    with pytest.raises(ModelBackendError) as caught:
        _adapter().generate_turn(
            GenerationRequest(
                messages=({"role": "user", "content": "produce one action"},),
                tools=({
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },),
                tool_choice="auto",
            )
        )

    assert calls == 1
    assert "typed boundary" in str(caught.value)
