from __future__ import annotations

from types import SimpleNamespace

import anyio

import minecraft_mod_ai.agent_capability_context as capability_context
from minecraft_mod_ai import agent_tool_runtime


def _schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_first_party_execution_does_not_repeat_schema_listing(monkeypatch) -> None:
    runtime = agent_tool_runtime.AgentToolRuntime(profile="test")
    events: list[str] = []

    class Session:
        async def list_tools(self):
            raise AssertionError("execution hot path repeated the stage schema listing")

        async def call_tool(self, name, *, arguments):
            events.append(f"execute:{name}:{arguments['query']}")
            return SimpleNamespace(
                isError=False,
                structuredContent={"ok": True},
                content=(),
            )

    class Context:
        async def __aenter__(self):
            events.append("enter")
            return Session()

        async def __aexit__(self, exc_type, exc, tb):
            events.append("exit")
            return False

    monkeypatch.setattr(runtime, "_session", lambda _stage: Context())
    result = anyio.run(
        runtime._call_tool_async,
        "generation",
        "search_code_rag",
        {"query": "block registry"},
    )

    assert result["structured_content"] == {"ok": True}
    assert events == ["enter", "execute:search_code_rag:block registry", "exit"]


def test_manifest_router_is_reused_but_request_target_stays_dynamic(monkeypatch) -> None:
    instances: list[object] = []
    targets: list[dict[str, str]] = []

    class FakeRouter:
        def __init__(self) -> None:
            instances.append(self)

        def capability_manifest(self, *, stage, target, max_access):
            targets.append(dict(target))
            return {"capabilities": {}}

    capability_context._manifest_router.cache_clear()
    monkeypatch.setattr(capability_context, "ExternalMCPRouter", FakeRouter)
    schemas = (_schema("external_mcp_capabilities"),)

    monkeypatch.setenv("MMM_MINECRAFT_VERSION", "1.21.1")
    capability_context.build_agent_capability_context(
        "research", schemas, model_role="researcher"
    )
    monkeypatch.setenv("MMM_MINECRAFT_VERSION", "1.21.4")
    capability_context.build_agent_capability_context(
        "research", schemas, model_role="researcher"
    )

    assert len(instances) == 1
    assert [target["minecraft_version"] for target in targets] == ["1.21.1", "1.21.4"]
    capability_context._manifest_router.cache_clear()
