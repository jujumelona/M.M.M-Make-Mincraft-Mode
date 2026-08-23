from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.forced_tool_execution_contract import (
    _arguments_match_schema,
    deterministic_forced_read_turn,
    install,
)


@dataclass(frozen=True)
class _Request:
    messages: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    tool_validation_schemas: tuple[dict[str, Any], ...] = ()
    tool_choice: object | None = None
    parallel_tool_calls: bool = True


def _schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
            },
        },
    }


def _project_rag_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_project_rag",
            "description": "exact-version project research",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "minecraft_version": {"type": "string", "minLength": 1},
                    "limit": {"type": "integer", "default": 6},
                },
                "required": ["query", "minecraft_version"],
            },
        },
    }


def _code_rag_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_code_rag",
            "description": "project code search",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"query": {"type": "string", "minLength": 1}},
                "required": ["query"],
            },
        },
    }


def test_local_forced_tool_uses_one_required_surface_then_protocol_retry() -> None:
    seen: list[_Request] = []

    class LocalAdapter:
        def generate_turn(self, request: _Request):
            seen.append(request)
            if len(seen) == 1:
                return SimpleNamespace(tool_calls=(), content="I think I am done")
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name="apply_source_edit"),),
                content="",
            )

    class RemoteAdapter:
        def generate_turn(self, request: _Request):
            return SimpleNamespace(tool_calls=(), content="unused")

    install(
        openai_compatible_module=SimpleNamespace(OpenAICompatibleAdapter=RemoteAdapter),
        llama_cpp_module=SimpleNamespace(LlamaCppAdapter=LocalAdapter),
    )

    edit = _schema("apply_source_edit")
    search = _schema("search_code_rag")
    request = _Request(
        messages=({"role": "user", "content": "repair"},),
        tools=(edit, search),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
    )

    turn = LocalAdapter().generate_turn(request)

    assert [call.name for call in turn.tool_calls] == ["apply_source_edit"]
    assert len(seen) == 2
    for constrained in seen:
        assert constrained.tools == (edit,)
        assert constrained.tool_choice == "required"
        assert constrained.parallel_tool_calls is False
    assert seen[0].messages == request.messages
    assert "previous assistant turn" in seen[1].messages[-1]["content"].casefold()


def test_local_forced_tool_returns_validation_only_stale_call_without_nested_retry() -> None:
    seen: list[_Request] = []

    class LocalAdapter:
        def generate_turn(self, request: _Request):
            seen.append(request)
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name="java_workspace_symbols"),),
                content="",
            )

    class RemoteAdapter:
        def generate_turn(self, request: _Request):
            return SimpleNamespace(tool_calls=(), content="unused")

    install(
        openai_compatible_module=SimpleNamespace(OpenAICompatibleAdapter=RemoteAdapter),
        llama_cpp_module=SimpleNamespace(LlamaCppAdapter=LocalAdapter),
    )

    current = _schema("search_project_rag")
    stale = _schema("java_workspace_symbols")
    request = _Request(
        messages=({"role": "user", "content": "repair"},),
        tools=(current,),
        tool_validation_schemas=(current, stale),
        tool_choice={
            "type": "function",
            "function": {"name": "search_project_rag"},
        },
    )

    turn = LocalAdapter().generate_turn(request)

    assert [call.name for call in turn.tool_calls] == ["java_workspace_symbols"]
    assert len(seen) == 1
    assert seen[0].tools == (current,)
    assert seen[0].tool_choice == "required"
    assert seen[0].messages == request.messages


def test_local_forced_project_rag_replaces_stale_write_with_host_read_call() -> None:
    seen: list[_Request] = []

    class LocalAdapter:
        def generate_turn(self, request: _Request):
            seen.append(request)
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name="apply_source_edit"),),
                content="",
            )

    class RemoteAdapter:
        def generate_turn(self, request: _Request):
            return SimpleNamespace(tool_calls=(), content="unused")

    install(
        openai_compatible_module=SimpleNamespace(OpenAICompatibleAdapter=RemoteAdapter),
        llama_cpp_module=SimpleNamespace(LlamaCppAdapter=LocalAdapter),
    )

    current = _project_rag_schema()
    stale = _schema("apply_source_edit")
    messages = (
        {
            "role": "user",
            "content": json.dumps(
                {
                    "phase": "implement_module",
                    "task": "Repair the approved registry module.",
                    "target": {"minecraft_version": "1.21.1"},
                    "module": {"module_id": "registry", "kind": "item"},
                }
            ),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-edit",
                    "type": "function",
                    "function": {
                        "name": "apply_source_edit",
                        "arguments": json.dumps(
                            {
                                "operation": "replace_exact",
                                "path": "src/main/java/example/Items.java",
                                "old": "secret source body",
                                "new": "replacement body",
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-edit",
            "name": "apply_source_edit",
            "content": json.dumps(
                {
                    "ok": False,
                    "error": "AgentToolRuntimeError: anchor mismatch api_key=do-not-leak",
                }
            ),
        },
    )
    request = _Request(
        messages=messages,
        tools=(current,),
        tool_validation_schemas=(current, stale),
        tool_choice={
            "type": "function",
            "function": {"name": "search_project_rag"},
        },
    )

    turn = LocalAdapter().generate_turn(request)

    assert len(seen) == 1
    assert [call.name for call in turn.tool_calls] == ["search_project_rag"]
    arguments = dict(turn.tool_calls[0].arguments)
    assert arguments["minecraft_version"] == "1.21.1"
    query = arguments["query"]
    assert "operation=replace_exact" in query
    assert "path=src/main/java/example/Items.java" in query
    assert "anchor mismatch" in query
    assert "do-not-leak" not in query
    assert "secret source body" not in query
    assert "replacement body" not in query


def test_local_forced_project_rag_does_not_fabricate_missing_platform_lock() -> None:
    seen: list[_Request] = []

    class LocalAdapter:
        def generate_turn(self, request: _Request):
            seen.append(request)
            return SimpleNamespace(
                tool_calls=(SimpleNamespace(name="apply_source_edit"),),
                content="",
            )

    class RemoteAdapter:
        def generate_turn(self, request: _Request):
            return SimpleNamespace(tool_calls=(), content="unused")

    install(
        openai_compatible_module=SimpleNamespace(OpenAICompatibleAdapter=RemoteAdapter),
        llama_cpp_module=SimpleNamespace(LlamaCppAdapter=LocalAdapter),
    )
    current = _project_rag_schema()
    stale = _schema("apply_source_edit")
    request = _Request(
        messages=({"role": "user", "content": "repair the project"},),
        tools=(current,),
        tool_validation_schemas=(current, stale),
        tool_choice={
            "type": "function",
            "function": {"name": "search_project_rag"},
        },
    )

    turn = LocalAdapter().generate_turn(request)

    assert len(seen) == 1
    assert [call.name for call in turn.tool_calls] == ["apply_source_edit"]


def test_deterministic_code_rag_needs_no_platform_version_and_never_synthesizes_write() -> None:
    search = _code_rag_schema()
    request = _Request(
        messages=({"role": "user", "content": "repair the registry source"},),
        tools=(search,),
        tool_validation_schemas=(search,),
        tool_choice={
            "type": "function",
            "function": {"name": "search_code_rag"},
        },
    )

    turn = deterministic_forced_read_turn(request, "search_code_rag")

    assert turn is not None
    assert [call.name for call in turn.tool_calls] == ["search_code_rag"]
    assert turn.tool_calls[0].arguments == {"query": "repair the registry source"}
    write_request = _Request(
        messages=request.messages,
        tools=(_schema("apply_source_edit"),),
        tool_choice={
            "type": "function",
            "function": {"name": "apply_source_edit"},
        },
    )
    assert deterministic_forced_read_turn(write_request, "apply_source_edit") is None


@pytest.mark.parametrize(
    "query_schema",
    (
        {"type": "string", "pattern": "^SAFE$"},
        {"const": "SAFE"},
        {"type": ["string", "null"], "const": "SAFE"},
        {"type": "string", "format": "uuid"},
        {"type": "string", "format": "mmm-unknown-format"},
    ),
)
def test_deterministic_read_fails_closed_on_complete_query_schema(
    query_schema: dict[str, Any],
) -> None:
    schema = _code_rag_schema()
    schema["function"]["parameters"]["properties"]["query"] = query_schema
    request = _Request(
        messages=({"role": "user", "content": "UNSAFE"},),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice={
            "type": "function",
            "function": {"name": "search_code_rag"},
        },
    )

    assert deterministic_forced_read_turn(request, "search_code_rag") is None


def test_host_arguments_obey_required_and_additional_properties() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {
                "anyOf": [
                    {"const": "SAFE"},
                    {"type": "string", "pattern": "^ALSO_SAFE$"},
                ]
            },
            "minecraft_version": {"type": "string"},
        },
        "required": ["query", "minecraft_version"],
    }

    assert _arguments_match_schema(
        {"query": "SAFE", "minecraft_version": "1.21.1"}, schema
    )
    assert not _arguments_match_schema({"query": "SAFE"}, schema)
    assert not _arguments_match_schema(
        {
            "query": "SAFE",
            "minecraft_version": "1.21.1",
            "unexpected": True,
        },
        schema,
    )


def test_writable_progress_stops_forcing_after_successful_source_mutation() -> None:
    seen: list[_Request] = []

    class Inner:
        def generate_turn(self, request: _Request):
            seen.append(request)
            return SimpleNamespace(tool_calls=(), content="implementation complete")

    messages = (
        {"role": "user", "content": json.dumps({"phase": "implement_module"})},
        {
            "role": "tool",
            "name": "apply_source_edit",
            "content": json.dumps(
                {
                    "ok": True,
                    "tool": "apply_source_edit",
                    "result": {
                        "schema_version": "mmm/source-patch-receipt-v1",
                        "status": "APPLIED",
                        "operations": [{"operation": "replace"}],
                    },
                }
            ),
        },
    )
    request = _Request(
        messages=messages,
        tools=(_schema("apply_source_edit"),),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    turn = _WritableProgressAdapter(Inner()).generate_turn(request)

    assert turn.content == "implementation complete"
    assert len(seen) == 1
    assert seen[0].tool_choice == "auto"
    assert seen[0].parallel_tool_calls is True
