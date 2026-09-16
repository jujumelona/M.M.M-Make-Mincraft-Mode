from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)
from minecraft_mod_ai.model_adapters.qwen_tool_parser import parse_qwen_tool_markup
from minecraft_mod_ai.model_tool_aliases import (
    canonical_model_tool,
    is_model_tool_alias,
)


def _tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {"type": "string", "enum": ["replace", "insert", "delete"]},
                    "content": {"type": "string"},
                },
                "required": ["path", "operation"],
                "additionalProperties": False,
            },
        },
    }


def _response(parameters: str, *, tool_name: str = "apply_source_edit"):
    text = f"<tool_call><function={tool_name}>{parameters}</function></tool_call>"
    request = GenerationRequest(tools=(_tool(),), tool_choice="required", parallel_tool_calls=False)
    return _native_tool_generation_response({"content": text}, request)


def _accepted(parameters: str, *, tool_name: str = "apply_source_edit"):
    response = _response(parameters, tool_name=tool_name)
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name != "__mmm_rejected_tool_call__"
    return call


def _rejected(parameters: str, failure_code: str):
    response = _response(parameters)
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "__mmm_rejected_tool_call__"
    assert call.arguments["failure_code"] == failure_code
    return call


def test_apply_source_edit_file_alias_normalizes_to_path() -> None:
    call = _accepted(
        "<parameter=file>src/main/java/example/Test.java</parameter>"
        "<parameter=action>replace</parameter>"
    )
    assert call.arguments == {"path": "src/main/java/example/Test.java", "operation": "replace"}


def test_apply_source_edit_apply_alias_normalizes_to_operation() -> None:
    call = _accepted(
        "<parameter=path>src/A.java</parameter><parameter=apply>replace</parameter>"
    )
    assert call.arguments == {"path": "src/A.java", "operation": "replace"}


def test_apply_source_edit_apply_object_wrapper_is_unwrapped() -> None:
    call = _accepted(
        '<parameter=apply>{"file":"src/A.java","action":"replace","content":"x"}</parameter>'
    )
    assert call.arguments == {"path": "src/A.java", "operation": "replace", "content": "x"}


def test_apply_source_edit_arguments_wrapper_is_unwrapped() -> None:
    call = _accepted(
        '<parameter=arguments>{"target_path":"src/A.java","op":"replace"}</parameter>'
    )
    assert call.arguments == {"path": "src/A.java", "operation": "replace"}


def test_nested_argument_wrappers_are_bounded_and_normalized() -> None:
    call = _accepted(
        '<parameter=arguments>{"params":{"file":"src/A.java","apply":"replace"}}</parameter>'
    )
    assert call.arguments == {"path": "src/A.java", "operation": "replace"}


def test_apply_source_edit_rejects_file_and_path_together() -> None:
    call = _rejected(
        "<parameter=file>src/A.java</parameter>"
        "<parameter=path>src/B.java</parameter>"
        "<parameter=operation>replace</parameter>",
        "TOOL_ARGUMENT_CONFLICT",
    )
    assert "canonical parameter 'path'" in call.arguments["error"]


def test_apply_source_edit_still_rejects_unknown_parameters() -> None:
    _rejected(
        "<parameter=path>src/A.java</parameter>"
        "<parameter=operation>replace</parameter>"
        "<parameter=bogus>nope</parameter>",
        "TOOL_SCHEMA_INVALID",
    )


def test_unknown_nested_parameter_is_not_dropped() -> None:
    _rejected(
        '<parameter=apply>{"file":"src/A.java","action":"replace","bogus":1}</parameter>',
        "TOOL_SCHEMA_INVALID",
    )


def test_recovered_operation_still_obeys_schema_enum() -> None:
    _rejected(
        "<parameter=path>src/A.java</parameter><parameter=apply>not-an-operation</parameter>",
        "TOOL_SCHEMA_INVALID",
    )


def test_enum_formatting_recovery_lives_in_admission() -> None:
    tool = _tool()
    tool["function"]["parameters"]["properties"]["operation"]["enum"] = ["replace_exact"]
    request = GenerationRequest(tools=(tool,), tool_choice="required", parallel_tool_calls=False)
    text = (
        "<tool_call><function=apply_source_edit>"
        "<parameter=path>src/A.java</parameter><parameter=action>replaceExact</parameter>"
        "</function></tool_call>"
    )
    response = _native_tool_generation_response({"content": text}, request)
    assert response.tool_calls[0].arguments["operation"] == "replace_exact"


def test_patch_file_alias_resolves_only_at_admission() -> None:
    call = _accepted(
        "<parameter=file>src/A.java</parameter>"
        "<parameter=action>replace</parameter>"
        "<parameter=content>x</parameter>",
        tool_name="patch_file",
    )
    assert call.name == "apply_source_edit"
    assert call.arguments == {"path": "src/A.java", "operation": "replace", "content": "x"}


def test_parser_preserves_patch_file_when_no_admission_surface_exists() -> None:
    text = (
        "<tool_call><function=patch_file>"
        "<parameter=file>src/A.java</parameter><parameter=action>replace</parameter>"
        "</function></tool_call>"
    )
    visible, calls = parse_qwen_tool_markup(text, {})
    assert visible == ""
    assert calls[0].name == "patch_file"
    assert calls[0].arguments == {"file": "src/A.java", "action": "replace"}


def test_patch_file_is_not_a_permission_alias() -> None:
    assert canonical_model_tool("patch_file") == "patch_file"
    assert not is_model_tool_alias("patch_file")
