from __future__ import annotations

from minecraft_mod_ai.model_adapters.base import GenerationRequest
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import (
    _native_tool_generation_response,
)


def _tool(*, include_workspace_root: bool = False) -> dict:
    properties = {"operation": {"type": "string"}, "path": {"type": "string"}}
    if include_workspace_root:
        properties["workspace_root"] = {"type": "string"}
    return {
        "type": "function",
        "function": {
            "name": "apply_source_edit",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": ["operation", "path"],
                "additionalProperties": False,
            },
        },
    }


def _response(parameters: str, *, include_workspace_root: bool = False):
    text = f"<tool_call><function=apply_source_edit>{parameters}</function></tool_call>"
    request = GenerationRequest(
        tools=(_tool(include_workspace_root=include_workspace_root),),
        tool_choice="required",
        parallel_tool_calls=False,
    )
    return _native_tool_generation_response({"content": text}, request)


def test_workspace_root_is_ignored_when_host_owned():
    response = _response(
        "<parameter=operation>replace</parameter>"
        "<parameter=path>src/Main.java</parameter>"
        "<parameter=workspace_root>/tmp/workspace</parameter>"
    )
    assert response.tool_calls[0].arguments == {"operation": "replace", "path": "src/Main.java"}


def test_nested_workspace_root_is_ignored_when_host_owned():
    response = _response(
        '<parameter=arguments>{"operation":"replace","path":"src/Main.java",'
        '"workspace_root":"/tmp/workspace"}</parameter>'
    )
    assert response.tool_calls[0].arguments == {"operation": "replace", "path": "src/Main.java"}


def test_unrelated_unknown_parameter_remains_rejected():
    response = _response(
        "<parameter=operation>replace</parameter>"
        "<parameter=path>src/Main.java</parameter>"
        "<parameter=random_root>/tmp/workspace</parameter>"
    )
    rejection = response.tool_calls[0]
    assert rejection.name == "__mmm_rejected_tool_call__"
    assert rejection.arguments["failure_code"] == "TOOL_SCHEMA_INVALID"


def test_workspace_root_is_preserved_when_schema_declares_it():
    response = _response(
        "<parameter=operation>replace</parameter>"
        "<parameter=path>src/Main.java</parameter>"
        "<parameter=workspace_root>/tmp/workspace</parameter>",
        include_workspace_root=True,
    )
    assert response.tool_calls[0].arguments == {
        "operation": "replace",
        "path": "src/Main.java",
        "workspace_root": "/tmp/workspace",
    }
