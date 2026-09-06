from __future__ import annotations

from types import SimpleNamespace

import minecraft_mod_ai.llama_structured_decode_policy as decode_policy
import minecraft_mod_ai.llama_tuning_pipeline as tuning_pipeline


def test_bounded_section_disables_thinking_without_touching_research_tools() -> None:
    class _Hardware:
        @staticmethod
        def _server_payload(adapter, request):
            payload = {
                "model": "local",
                "messages": list(request.messages),
                "max_tokens": adapter.config.max_new_tokens,
            }
            if request.tools:
                payload["tools"] = list(request.tools)
            if request.response_format == "json":
                payload["response_format"] = {"type": "json_object"}
            return payload

    decode_policy.bind_structured_decode_policy(_Hardware)
    adapter = SimpleNamespace(config=SimpleNamespace(max_new_tokens=8192))

    section_request = SimpleNamespace(
        messages=({"role": "user", "content": "serialize section"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"section": {"type": "object"}},
        },
        tools=(),
    )
    section_payload = _Hardware._server_payload(adapter, section_request)
    assert section_payload["thinking_budget_tokens"] == 0
    assert section_payload["reasoning_effort"] == "none"

    research_request = SimpleNamespace(
        messages=({"role": "user", "content": "research"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"research_note": {"type": "object"}},
        },
        tools=(
            {
                "type": "function",
                "function": {"name": "search_project_rag", "parameters": {}},
            },
        ),
    )
    research_payload = _Hardware._server_payload(adapter, research_request)
    assert "thinking_budget_tokens" not in research_payload
    assert "reasoning_effort" not in research_payload
    assert research_payload["tools"]

    generic_json_request = SimpleNamespace(
        messages=({"role": "user", "content": "other json"},),
        response_format="json",
        response_schema={
            "type": "object",
            "properties": {"game_design": {"type": "object"}},
        },
        tools=(),
    )
    generic_payload = _Hardware._server_payload(adapter, generic_json_request)
    assert "thinking_budget_tokens" not in generic_payload
    assert "reasoning_effort" not in generic_payload


def test_native_tuning_pipeline_keeps_single_graph_owned_stage_order() -> None:
    pipeline = tuning_pipeline.NativeLlamaTuningPipeline(
        autotune=SimpleNamespace(),
        hardware_policy=SimpleNamespace(),
        runtime_tuning=SimpleNamespace(),
    )
    assert tuple(stage.name for stage in pipeline.stages()) == (
        "runtime-types",
        "hardware",
        "efficiency",
        "runtime",
        "cache-reuse",
        "decode-speed",
        "kernel-autotune",
        "qwen-transport",
        "multimodal",
    )
