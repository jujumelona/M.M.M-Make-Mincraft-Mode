from __future__ import annotations

from typing import Any

from minecraft_mod_ai import causal_frontier_adapter as frontier_module
from minecraft_mod_ai.causal_frontier_adapter import CausalFrontierAdapter
from minecraft_mod_ai.model_adapters import GenerationRequest, GenerationResponse


def _schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


class _InnerAdapter:
    def __init__(self) -> None:
        self.request: GenerationRequest | None = None

    def generate_turn(self, request: GenerationRequest) -> GenerationResponse:
        self.request = request
        return GenerationResponse(content="ok")


def test_per_turn_frontier_freezes_runtime_state_before_graph_search(monkeypatch: Any) -> None:
    schemas = (_schema("search_code_rag"), _schema("apply_source_patch"))
    observed: dict[str, Any] = {}

    def fake_frontier(
        candidates: Any,
        *,
        state: Any,
        goals: Any,
        limit: int,
        max_depth: int,
        preference: Any,
    ) -> tuple[str, ...]:
        del candidates, goals, limit, max_depth, preference
        observed["state"] = state
        assert isinstance(state, frozenset)
        return ("search_code_rag",)

    monkeypatch.setattr(frontier_module, "authorized_tools", lambda fallback: tuple(fallback))
    monkeypatch.setattr(frontier_module, "executable_frontier", fake_frontier)
    monkeypatch.setattr(
        frontier_module,
        "verified_state_from_messages",
        lambda *args, **kwargs: frozenset({"workspace_bound"}),
    )
    monkeypatch.setattr(
        frontier_module,
        "host_baseline_causal_facts",
        lambda messages: {"workspace_bound"},
    )
    monkeypatch.setattr(
        frontier_module,
        "_with_capability_context",
        lambda messages, **kwargs: tuple(dict(message) for message in messages),
    )

    inner = _InnerAdapter()
    adapter = CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=False,
    )
    request = GenerationRequest(
        messages=({"role": "user", "content": "Implement the approved module."},),
        tools=schemas,
        tool_choice="auto",
    )

    response = adapter.generate_turn(request)

    assert response.content == "ok"
    assert observed["state"] == frozenset({"workspace_bound"})
    assert inner.request is not None
    assert tuple(
        schema["function"]["name"] for schema in inner.request.tools
    ) == ("search_code_rag",)
