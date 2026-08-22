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


def _search_schema(name: str) -> dict:
    schema = _schema(name)
    properties = {"query": {"type": "string", "minLength": 1}}
    required = ["query"]
    if name == "search_project_rag":
        properties["minecraft_version"] = {"type": "string", "minLength": 1}
        required.append("minecraft_version")
    schema["function"]["parameters"] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return schema


def _turn(name: str):
    return SimpleNamespace(tool_calls=(SimpleNamespace(name=name),))


def test_runtime_stale_recovery_reaches_prebound_coder_alias() -> None:
    canonical = causal_frontier_adapter.CausalFrontierAdapter

    assert coder_tool_route_integrity_contract.CausalFrontierAdapter is canonical
    assert causal_tool_frontier_contract.CausalFrontierAdapter is canonical
    assert recovery.is_installed()


def test_stale_recovery_forces_one_current_tool_once(monkeypatch) -> None:
    requests = []

    class Inner:
        def generate_turn(self, request):
            requests.append(request)
            return _turn("search_project_rag")

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

    assert result.tool_calls[0].name == "search_project_rag"
    assert len(requests) == 1
    retry = requests[0]
    assert retry.tool_choice["function"]["name"] == "search_project_rag"
    assert len(retry.tools) == 1
    assert retry.tools[0]["function"]["name"] == "search_project_rag"
    assert adapter.published == [("search_project_rag",)]
    assert adapter.reset_count >= 2


def test_stale_recovery_prefers_host_read_without_project_version_or_decode(
    monkeypatch,
) -> None:
    requests = []

    class Inner:
        def generate_turn(self, request):
            requests.append(request)
            return _turn("apply_source_edit")

    class FakeAdapter:
        def __init__(self) -> None:
            self.inner = Inner()
            self.stage = "generation"
            self.role = "coder"
            self.authorized_surface = (
                _search_schema("search_project_rag"),
                _schema("java_workspace_symbols"),
                _search_schema("search_code_rag"),
                _schema("apply_source_edit"),
            )
            self.published = []

        def _reset_stale_guard(self) -> None:
            return None

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
        messages=({"role": "user", "content": "repair current Java source"},),
        tools=adapter.authorized_surface,
        tool_validation_schemas=adapter.authorized_surface,
        tool_choice="auto",
        parallel_tool_calls=True,
        metadata={"target": {"minecraft_version": "1.20.1"}},
    )

    result = adapter.generate_turn(request)

    assert [call.name for call in result.tool_calls] == ["search_code_rag"]
    assert result.tool_calls[0].arguments == {"query": "repair current Java source"}
    assert requests == []
    assert adapter.published == [("search_code_rag",)]


def test_non_mutation_stale_recovery_keeps_constructible_forced_project_rag(
    monkeypatch,
) -> None:
    requests = []

    class Inner:
        def generate_turn(self, request):
            requests.append(request)
            return _turn("java_workspace_symbols")

    class FakeAdapter:
        def __init__(self) -> None:
            self.inner = Inner()
            self.stage = "generation"
            self.role = "coder"
            self.authorized_surface = (
                _search_schema("search_project_rag"),
                _search_schema("search_code_rag"),
                _schema("java_workspace_symbols"),
            )
            self.published = []

        def _reset_stale_guard(self) -> None:
            return None

        def _publish_frontier(self, names) -> None:
            self.published.append(tuple(names))

        def generate_turn(self, request):
            return _turn("java_workspace_symbols")

    monkeypatch.setattr(
        recovery,
        "current_frontier_names",
        lambda: ("search_project_rag", "search_code_rag"),
    )
    recovery._install_generate_turn(FakeAdapter)
    adapter = FakeAdapter()
    request = GenerationRequest(
        messages=({"role": "user", "content": "inspect target APIs"},),
        tools=adapter.authorized_surface,
        tool_validation_schemas=adapter.authorized_surface,
        tool_choice="auto",
        parallel_tool_calls=True,
        metadata={"target": {"minecraft_version": "1.20.1"}},
    )

    result = adapter.generate_turn(request)

    assert [call.name for call in result.tool_calls] == ["search_project_rag"]
    assert result.tool_calls[0].arguments == {
        "query": "inspect target APIs",
        "minecraft_version": "1.20.1",
    }
    assert requests == []
    assert adapter.published == [("search_project_rag",)]
