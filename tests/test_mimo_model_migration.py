from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_registry import ModelRegistry
from minecraft_mod_ai.llama_multimodal_contract import _requires_media_baseline
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _native_tool_generation_response
from minecraft_mod_ai.llama_server_runtime_tuning import _candidate_variants_for_config
from minecraft_mod_ai.llama_lora_runtime import configured_lora_specs
from minecraft_mod_ai.llama_server_hardware_policy import _server_payload
from minecraft_mod_ai.llama_server_autotune import _assert_mimo_server_compatible


def test_default_t4_mimo_has_no_legacy_qwen_lora_or_mtp() -> None:
    for profile_name in ("t4_local", "t4_quality", "MiMo-V2.6-9B_6GB"):
        profile = ModelRegistry().load_profile(profile_name)
        for role in ("planner", "researcher", "coder", "coder_safe", "visual_critic"):
            config = profile.roles[role]
            assert config.model_id == "bartowski/MiMo-V2.6-Distill-Qwen-9B-GGUF"
            assert config.extra["gguf_filename"] == "MiMo-V2.6-Distill-Qwen-9B-Q4_K_M.gguf"
            assert config.extra["mmproj_filename"] == "mmproj-MiMo-V2.6-Distill-Qwen-9B-f16.gguf"
            assert config.extra["runtime_contract"] == "mimo"
            assert config.extra["supports_mtp"] is False
            assert "decode_hotpath" not in config.extra
            assert "lora_adapters" not in config.extra
            assert configured_lora_specs(config) == ()
            from minecraft_mod_ai import llama_server_autotune
            variants = _candidate_variants_for_config(llama_server_autotune, config)
            assert all(variant.spec_type != "draft-mtp" for variant in variants)
            assert "qwen_family" not in config.extra
            assert _requires_media_baseline(config)


def _tool_request() -> GenerationRequest:
    return GenerationRequest(
        messages=({"role": "user", "content": "Look up the item"},),
        tools=({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find an item",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                    "additionalProperties": False,
                },
            },
        },),
        tool_choice="required",
        parallel_tool_calls=False,
    )


def test_mimo_uses_native_tool_calls_not_legacy_qwen_markup() -> None:
    request = _tool_request()
    message = {
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": "call_mimo_1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"q":"diamond"}'},
        }],
    }
    turn = _native_tool_generation_response(message, request, runtime_contract="mimo")
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "lookup"
    assert turn.tool_calls[0].arguments == {"q": "diamond"}
    raw_xml = "<tool_call><function=lookup><parameter=q>diamond</parameter></function></tool_call>"
    with pytest.raises(RuntimeError, match="MIMO_NATIVE_TOOL_CALLS_REQUIRED"):
        _native_tool_generation_response({"content": raw_xml}, request, runtime_contract="mimo")


def test_mimo_thinking_and_tool_payload_follow_native_template() -> None:
    model = ModelRegistry().role("t4_local", "planner")
    adapter = SimpleNamespace(config=model)
    action = _server_payload(adapter, _tool_request())
    assert action["chat_template_kwargs"] == {"enable_thinking": False}
    assert action["tool_choice"] == "required"
    plain = _server_payload(
        adapter, GenerationRequest(messages=({"role": "user", "content": "Hello"},))
    )
    assert plain["chat_template_kwargs"] == {"enable_thinking": True}



def test_mimo_old_server_fails_closed(monkeypatch) -> None:
    from minecraft_mod_ai import llama_server_autotune
    config = SimpleNamespace(extra={"runtime_contract": "mimo"})
    monkeypatch.setattr(llama_server_autotune, "_server_version", lambda _: "version: 11101 (old)")
    with pytest.raises(RuntimeError, match="b11102"):
        _assert_mimo_server_compatible("llama-server", config)
    monkeypatch.setattr(llama_server_autotune, "_server_version", lambda _: "version: 11429 (new)")
    _assert_mimo_server_compatible("llama-server", config)
