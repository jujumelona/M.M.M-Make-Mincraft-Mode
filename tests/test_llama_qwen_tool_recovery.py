from __future__ import annotations

import pytest

from minecraft_mod_ai.model_adapters import llama_cpp_adapter as llama
from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.qwen_tool_parser import ToolCallValidationError
from minecraft_mod_ai.source_edit_scalar_protocol_contract import SOURCE_EDIT_SCHEMA


def _request(*, parallel: bool = True, choice=None) -> GenerationRequest:
    tool = {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "description": "Apply one source edit",
            "parameters": SOURCE_EDIT_SCHEMA,
        },
    }
    return GenerationRequest(
        tools=(tool,),
        tool_choice=choice,
        parallel_tool_calls=parallel,
    )


def test_recovers_observed_qwen_payload_markup_when_native_calls_are_empty() -> None:
    message = {
        "tool_calls": [],
        "content": """<tool_call>
<function=apply_source_edit>
<parameter=payload>{"operation":"create_file","path":"src/main/resources/debug-token.txt","content":"deterministic-debug-token"}</parameter>
</function>
</tool_call>""",
    }

    response = llama._native_tool_generation_response(message, _request())

    assert response.content == ""
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "apply_source_edit"
    assert call.arguments == {
        "operation": "create_file",
        "path": "src/main/resources/debug-token.txt",
        "content": "deterministic-debug-token",
    }


def test_native_tool_calls_remain_authoritative_over_content_markup() -> None:
    message = {
        "tool_calls": [
            {
                "id": "native-1",
                "type": "function",
                "function": {
                    "name": "apply_source_edit",
                    "arguments": {
                        "operation": "create",
                        "path": "src/main/resources/native.txt",
                        "text": "native",
                    },
                },
            }
        ],
        "content": """<tool_call>
<function=apply_source_edit>
<parameter=payload>{"operation":"create_file","path":"src/main/resources/fallback.txt","content":"fallback"}</parameter>
</function>
</tool_call>""",
    }

    response = llama._native_tool_generation_response(message, _request())

    assert [call.id for call in response.tool_calls] == ["native-1"]
    assert response.tool_calls[0].arguments["path"] == "src/main/resources/native.txt"


def test_plain_prose_is_not_reconstructed_as_a_tool_call_for_auto_choice() -> None:
    message = {"tool_calls": [], "content": "I would edit the file next."}

    response = llama._native_tool_generation_response(message, _request(choice="auto"))

    assert response.tool_calls == ()
    assert response.content == "I would edit the file next."


def test_required_choice_rejects_plain_prose_without_tool_call() -> None:
    message = {"tool_calls": [], "content": "I would edit the file next."}

    with pytest.raises(ToolCallValidationError, match="required"):
        llama._native_tool_generation_response(message, _request(choice="required"))


def test_schema_invalid_markup_is_not_silently_downgraded_to_prose() -> None:
    message = {
        "tool_calls": [],
        "content": """<tool_call>
<function=apply_source_edit>
<parameter=payload>{"operation":"append","path":"src/main/resources/debug-token.txt","content":"bad"}</parameter>
</function>
</tool_call>""",
    }

    with pytest.raises(ToolCallValidationError, match="outside enum"):
        llama._native_tool_generation_response(message, _request())
