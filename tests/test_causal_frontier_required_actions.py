from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.causal_frontier_adapter import CausalFrontierAdapter
from minecraft_mod_ai.model_adapters.base import GenerationRequest, GenerationResponse


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


class _Inner:
    def __init__(self) -> None:
        self.request = None

    def generate_turn(self, request):
        self.request = request
        return GenerationResponse(content="")


class _Ledger:
    def resolve(self, *args, **kwargs):
        return SimpleNamespace(
            state=frozenset({"code_evidence", "evidence_ready", "workspace_bound"}),
            query="fix the implementation",
            blocked_mutation_tools=frozenset(),
        )


def test_single_mutation_frontier_becomes_named_required_action(monkeypatch):
    source_edit = _tool("apply_source_edit")
    inner = _Inner()
    adapter = CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=True,
        authorized_surface=(source_edit,),
    )
    adapter._state_ledger = _Ledger()
    monkeypatch.setattr(
        "minecraft_mod_ai.causal_frontier_adapter.executable_frontier",
        lambda *args, **kwargs: ("apply_source_edit",),
    )

    request = GenerationRequest(
        messages=({"role": "user", "content": "fix the implementation"},),
        tools=(source_edit,),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    adapter.generate_turn(request)

    assert inner.request.tool_choice == {
        "type": "function",
        "function": {"name": "apply_source_edit"},
    }
    assert inner.request.parallel_tool_calls is False


def test_read_only_single_frontier_stays_auto(monkeypatch):
    search = _tool("search_code_rag")
    inner = _Inner()
    adapter = CausalFrontierAdapter(
        inner,
        stage="generation",
        role="coder",
        require_fresh_evidence=True,
        authorized_surface=(search,),
    )
    adapter._state_ledger = _Ledger()
    monkeypatch.setattr(
        "minecraft_mod_ai.causal_frontier_adapter.executable_frontier",
        lambda *args, **kwargs: ("search_code_rag",),
    )

    request = GenerationRequest(
        messages=({"role": "user", "content": "inspect the implementation"},),
        tools=(search,),
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    adapter.generate_turn(request)

    assert inner.request.tool_choice == "auto"
    assert inner.request.parallel_tool_calls is False
