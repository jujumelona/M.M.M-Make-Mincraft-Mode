from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import causal_frontier_adapter
from minecraft_mod_ai import causal_stale_tool_recovery_contract as recovery
from minecraft_mod_ai import causal_tool_frontier_contract
from minecraft_mod_ai import coder_tool_route_integrity_contract
from minecraft_mod_ai.model_adapters.base import GenerationRequest


def _schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _turn(name: str):
    return SimpleNamespace(tool_calls=(SimpleNamespace(name=name),))


def test_runtime_stale_recovery_reaches_prebound_coder_alias() -> None:
    canonical = causal_frontier_adapter.CausalFrontierAdapter

    assert coder_tool_route_integrity_contract.CausalFrontierAdapter is canonical
    assert causal_tool_frontier_contract.CausalFrontierAdapter is canonical
    assert getattr(canonical, "_mmm_stale_tool_recovery_v1", False)
    assert getattr(canonical.generate_turn, "_mmm_stale_tool_recovery_v1", False)


def test_stale_recovery_forces_one_current_tool_and_rotates(monkeypatch) -> None:
    requests = []

    class Inner:
        def __init__(self) -> None:
            self.outputs = iter(
                (
                    _turn("apply_source_edit"),
                    _turn("apply_source_edit"),
                    _turn("search_code_rag"),
                )
            )

        def generate_turn(self, request):
            requests.append(request)
            return next(self.outputs)

    class FakeAdapter:
        def __init__(self) -> None:
            self.inner = Inner()
            self.stage = "generation"
            self.role = "coder"
            self.authorized_surface = (
                _schema("search_project_rag"),
                _schema("java_workspace_symbols"),
                _schema("search_code_rag"),
                _schema("apply_source_edit"),
            )
            self.published = []
            self.reset_count = 0

        def _reset_stale_guard(self) -> None:
            self.reset_count += 1

        def _publish_frontier(self, names) -> None:
            self.published.append(tuple(names))

        def generate_turn(self, request):
            return _turn("apply_source_edit")

    monkeypatch.setattr(
        recovery,
        "current_frontier_names",
        lambda: (
            "search_project_rag",
            "java_workspace_symbols",
            "search_code_rag",
        ),
    )
    recovery._install_generate_turn(FakeAdapter)
    adapter = FakeAdapter()
    request = GenerationRequest(
        messages=({"role": "user", "content": "repair"},),
        tools=adapter.authorized_surface,
        tool_validation_schemas=adapter.authorized_surface,
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    result = adapter.generate_turn(request)

    assert result.tool_calls[0].name == "search_code_rag"
    assert len(requests) == 3
    forced_names = [row.tool_choice["function"]["name"] for row in requests]
    assert forced_names == [
        "search_project_rag",
        "java_workspace_symbols",
        "search_code_rag",
    ]
    assert all(len(row.tools) == 1 for row in requests)
    assert [row.tools[0]["function"]["name"] for row in requests] == forced_names
    assert adapter.published == [
        ("search_project_rag",),
        ("java_workspace_symbols",),
        ("search_code_rag",),
    ]
    assert adapter.reset_count >= 2
