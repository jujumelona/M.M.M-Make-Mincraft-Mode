from __future__ import annotations

import json

from minecraft_mod_ai.causal_tool_graph import executable_frontier
from minecraft_mod_ai.coder_tool_route_integrity_contract import _WritableProgressAdapter
from minecraft_mod_ai.model_adapters import (
    GenerationRequest,
    GenerationResponse,
    ToolCall,
)


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _messages(*failures: str) -> tuple[dict[str, object], ...]:
    messages: list[dict[str, object]] = [
        {
            "role": "user",
            "content": json.dumps({"phase": "implement_module"}),
        }
    ]
    for index, name in enumerate(failures):
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"failure-{index}",
                "name": name,
                "content": json.dumps(
                    {"ok": False, "tool": name, "error": "synthetic failure"}
                ),
            }
        )
    return tuple(messages)


class _RecordingAdapter:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        choice = request.tool_choice
        assert isinstance(choice, dict)
        function = choice.get("function")
        assert isinstance(function, dict)
        name = str(function.get("name", ""))
        return GenerationResponse(
            tool_calls=(ToolCall(id="call-1", name=name),)
        )


def test_direct_repair_frontier_prefers_minimum_cost_edit() -> None:
    frontier = executable_frontier(
        (_tool("apply_source_edit"), _tool("apply_source_patch")),
        state=frozenset({"workspace_bound", "project_observed", "evidence_ready"}),
        goals=("repair",),
    )

    assert frontier == ("apply_source_edit",)


def test_writable_coder_prefers_hardened_patch_over_schema_order() -> None:
    inner = _RecordingAdapter()
    adapter = _WritableProgressAdapter(inner)
    request = GenerationRequest(
        messages=_messages(),
        tools=(_tool("apply_source_edit"), _tool("apply_source_patch")),
        tool_choice="auto",
    )

    adapter.generate_turn(request)

    assert inner.requests[0].tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_patch"},
    }


def test_writable_coder_does_not_locally_fail_over_after_patch_failures() -> None:
    inner = _RecordingAdapter()
    adapter = _WritableProgressAdapter(inner)
    request = GenerationRequest(
        messages=_messages("apply_source_patch", "apply_source_patch"),
        tools=(_tool("apply_source_edit"), _tool("apply_source_patch")),
        tool_choice="auto",
    )

    adapter.generate_turn(request)

    assert inner.requests[0].tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_patch"},
    }


def test_writable_coder_does_not_own_retry_exhaustion() -> None:
    inner = _RecordingAdapter()
    adapter = _WritableProgressAdapter(inner)
    request = GenerationRequest(
        messages=_messages(
            "apply_source_patch",
            "apply_source_patch",
            "apply_source_edit",
            "apply_source_edit",
        ),
        tools=(_tool("apply_source_edit"), _tool("apply_source_patch")),
        tool_choice="auto",
    )

    adapter.generate_turn(request)

    assert len(inner.requests) == 1
    assert inner.requests[0].tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_patch"},
    }
