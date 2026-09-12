from __future__ import annotations

import json

from minecraft_mod_ai import forced_tool_execution_contract as forced_contract
from minecraft_mod_ai import llama_stream_efficiency_contract as stream_contract
from minecraft_mod_ai.model_adapters import llama_cpp_adapter
from minecraft_mod_ai.model_adapters.base import AdapterConfig, GenerationRequest


def _source_edit_schema() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one source edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["operation", "path", "content"],
                "additionalProperties": False,
            },
        },
    }


def test_valid_first_source_mutation_completion_uses_one_low_level_completion(monkeypatch) -> None:
    completion_calls = 0
    arguments = {
        "operation": "create_file",
        "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
        "content": "package dev.mmm.debugfixture;\npublic final class DebugToken {}\n",
    }

    def fake_completion(*_args, **_kwargs):
        nonlocal completion_calls
        completion_calls += 1
        return {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_direct_source_edit",
                    "type": "function",
                    "function": {
                        "name": "apply_source_edit",
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        }

    monkeypatch.setattr(
        llama_cpp_adapter,
        "_completion_message_with_prefill",
        fake_completion,
    )
    monkeypatch.setattr(
        llama_cpp_adapter,
        "_tool_server_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        stream_contract,
        "_report_server_connection",
        lambda *_args, **_kwargs: None,
    )

    class DirectFirstLlama(llama_cpp_adapter.LlamaCppAdapter):
        def _server_url(self, _request):
            return "http://unit-test.invalid"

    forced_contract._install_adapter_class(
        DirectFirstLlama,
        transport_name="test llama",
        deterministic_stale_read=True,
    )
    adapter = DirectFirstLlama(
        AdapterConfig(
            role="coder",
            adapter="llama_cpp",
            model_id="unit-test",
        )
    )
    schema = _source_edit_schema()
    request = GenerationRequest(
        messages=({"role": "user", "content": "Create DebugToken.java"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
        parallel_tool_calls=False,
    )

    turn = adapter.generate_turn(request)

    assert completion_calls == 1
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].name == "apply_source_edit"
    assert dict(turn.tool_calls[0].arguments) == arguments
