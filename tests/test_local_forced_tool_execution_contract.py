from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.forced_tool_execution_contract import install


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


def test_local_forced_tool_uses_prompt_and_auto_transport_then_recovers_none() -> None:
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
        assert constrained.tool_choice == "auto"
        assert constrained.parallel_tool_calls is False
    assert "host requires" in seen[0].messages[-1]["content"].casefold()
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
    assert seen[0].tool_choice == "auto"
    assert "previous assistant turn" not in seen[0].messages[-1]["content"].casefold()


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
                    "result": {"status": "APPLIED"},
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
