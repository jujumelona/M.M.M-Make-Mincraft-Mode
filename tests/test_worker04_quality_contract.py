from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import MethodType

import minecraft_mod_ai.agent_capability_context as capability_context
from minecraft_mod_ai import external_mcp_router
from minecraft_mod_ai.external_agent_bridge import ExternalAgentBridge
from minecraft_mod_ai.external_mcp_router import ExternalMCPRouter
from minecraft_mod_ai.model_adapters.base import ToolCall
from minecraft_mod_ai.model_router import _execute_tool_waves, _parallel_read_call


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_request_preparation_reads_role_policy_once(monkeypatch) -> None:
    original = capability_context.load_agent_role_routes
    calls = 0

    def counted_routes():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(capability_context, "load_agent_role_routes", counted_routes)
    tools, context = capability_context.prepare_agent_tool_surface(
        "generation",
        "coder",
        (_schema("search_code_rag"), _schema("search_project_rag")),
    )

    assert calls == 1
    assert {schema["function"]["name"] for schema in tools} == {
        "search_code_rag",
        "search_project_rag",
    }
    assert '"model_role":"coder"' in context


def test_read_only_external_mcp_calls_execute_in_one_parallel_wave() -> None:
    calls = (
        ToolCall(id="one", name="external_mcp_call", arguments={"max_access": "read"}),
        ToolCall(id="two", name="external_mcp_call", arguments={"max_access": "read"}),
    )
    entered = threading.Barrier(2)

    def execute(call: ToolCall):
        entered.wait(timeout=1.0)
        return call, {"ok": True}

    result = _execute_tool_waves(calls, execute)
    assert [call.id for call, _ in result] == ["one", "two"]


def test_external_mcp_parallel_classification_never_promotes_mutations() -> None:
    assert _parallel_read_call(
        ToolCall(id="read", name="external_mcp_call", arguments={"max_access": "read"})
    )
    assert _parallel_read_call(
        ToolCall(id="default-read", name="external_mcp_call", arguments={})
    )
    assert not _parallel_read_call(
        ToolCall(id="write", name="external_mcp_call", arguments={"max_access": "write"})
    )
    assert not _parallel_read_call(
        ToolCall(id="admin", name="external_mcp_call", arguments={"max_access": "admin"})
    )


def test_external_router_provider_bridge_has_no_router_wide_io_lock() -> None:
    router = ExternalMCPRouter(timeout_seconds=2.0)
    entered = threading.Barrier(2)

    async def fake_provider(self, server_name, entry, *, tool, arguments):
        entered.wait(timeout=1.0)
        return {"server_info": {"name": server_name}, "result": {"ok": True}}

    router._call_provider_async = MethodType(fake_provider, router)

    def invoke(index: int):
        return router._call_provider(
            f"server-{index}",
            {},
            tool="read",
            arguments={},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(invoke, range(2)))
    assert all(result["result"]["ok"] for result in results)


def test_external_bridge_lazy_router_initializes_once_under_concurrency(monkeypatch) -> None:
    created: list[object] = []
    launch = threading.Barrier(4)

    class FakeRouter:
        def __init__(self, *, timeout_seconds: float) -> None:
            created.append(self)
            self.timeout_seconds = timeout_seconds

    monkeypatch.setattr(external_mcp_router, "ExternalMCPRouter", FakeRouter)
    bridge = ExternalAgentBridge(timeout_seconds=2.0)

    def resolve(_: int):
        launch.wait(timeout=1.0)
        return bridge._external_router()

    with ThreadPoolExecutor(max_workers=4) as executor:
        routers = list(executor.map(resolve, range(4)))

    assert len(created) == 1
    assert all(router is routers[0] for router in routers)
