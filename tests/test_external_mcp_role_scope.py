from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime
from minecraft_mod_ai.external_agent_bridge import (
    CALL_TOOL,
    CAPABILITIES_TOOL,
    SCHEMA_TOOL,
    ExternalAgentBridge,
)
from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.external_mcp_router import ExternalMCPRouter
from minecraft_mod_ai.mcp_schema_integrity_contract import ensure_schema_environment
from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter


def _registry(tmp_path: Path) -> ExternalMCPRegistry:
    path = tmp_path / "role-scope-mcp.yaml"
    path.write_text(
        """schema_version: mmm/external-mcp-registry-v2
servers:
  allowed-provider:
    status: enabled
    transport: stdio
    command: [allowed]
    version_policy: dynamic
    loaders: [fabric]
    trust: test
    capabilities:
      source_search:
        tool: allowed_search
        access: read
        stages: [generation]
        priority: 20
  forbidden-provider:
    status: enabled
    transport: stdio
    command: [forbidden]
    version_policy: dynamic
    loaders: [fabric]
    trust: test
    capabilities:
      source_search:
        tool: forbidden_search
        access: read
        stages: [generation]
        priority: 1
      forbidden_only:
        tool: forbidden_only
        access: read
        stages: [generation]
        priority: 1
""",
        encoding="utf-8",
    )
    return ExternalMCPRegistry(path)


def test_router_filters_manifest_and_invocation_to_role_server_scope(tmp_path, monkeypatch) -> None:
    router = ExternalMCPRouter(_registry(tmp_path))
    manifest = router.capability_manifest(
        stage="generation", allowed_server_ids={"allowed-provider"}
    )
    assert set(manifest["capabilities"]) == {"source_search"}
    assert [row["server"] for row in manifest["capabilities"]["source_search"]] == [
        "allowed-provider"
    ]
    calls: list[str] = []

    def fake_call(server_name, entry, *, tool, arguments):
        calls.append(server_name)
        return {
            "server_info": {"name": server_name},
            "result": {"structured": {"hits": [server_name]}},
        }

    monkeypatch.setattr(router, "_call_provider", fake_call)
    bundle = router.invoke(
        "source_search",
        stage="generation",
        arguments={"query": "Block"},
        allowed_server_ids={"allowed-provider"},
    )
    assert bundle["status"] == "PASS"
    assert calls == ["allowed-provider"]
    assert bundle["evidence"][0]["server"] == "allowed-provider"
    denied = router.invoke(
        "forbidden_only",
        stage="generation",
        allowed_server_ids={"allowed-provider"},
    )
    assert denied["status"] == "UNAVAILABLE"
    assert denied["attempts"] == []


def test_bridge_schema_binding_isolated_by_server_scope_and_owner_reused(monkeypatch) -> None:
    bridge = ExternalAgentBridge()
    selected: list[frozenset[str]] = []
    revalidated: list[str] = []

    class ScopeRegistry:
        def routes(self, capability, **kwargs):
            assert capability == "source_search"
            return [
                {
                    "server": server,
                    "entry": {
                        "status": "enabled",
                        "transport": "stdio",
                        "command": [server],
                        "trust": "test",
                    },
                    "route": {
                        "tool": f"{server}-lookup",
                        "access": "read",
                        "target_args": {},
                    },
                }
                for server in ("provider-a", "provider-b")
            ]

    class ScopeRouter:
        def __init__(self) -> None:
            self.registry = ScopeRegistry()

        @staticmethod
        def _configured(entry):
            return entry.get("status") == "enabled"

        @staticmethod
        def _child_env(entry):
            return {}

        @staticmethod
        def _server_url(entry):
            return ""

    bridge._router = ScopeRouter()

    async def fake_describe(stage, capability, target, max_access, allowed_server_ids):
        scope = frozenset(allowed_server_ids or ())
        selected.append(scope)
        server = sorted(scope)[0]
        return {
            "schema_version": "fixture/external-mcp-schema-v1",
            "capability": capability,
            "stage": stage,
            "target": dict(target),
            "server": server,
            "tool": f"{server}-lookup",
            "access": max_access,
            "trust": "test",
            "target_args_injected_by_router": {},
            "description": "fixture lookup",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "status": "PASS",
        }

    async def fake_provider_schema(entry, *, tool, env, url, timeout_seconds):
        revalidated.append(tool)
        return {
            "description": "fixture lookup",
            "input_schema": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        }

    monkeypatch.setattr(bridge, "_describe_async", fake_describe)
    monkeypatch.setattr(
        "minecraft_mod_ai.external_agent_bridge._provider_schema",
        fake_provider_schema,
    )
    one = bridge.call(
        "generation",
        SCHEMA_TOOL,
        {"capability": "source_search"},
        allowed_server_ids={"provider-a"},
    )
    two = bridge.call(
        "generation",
        SCHEMA_TOOL,
        {"capability": "source_search"},
        allowed_server_ids={"provider-b"},
    )
    again = bridge.call(
        "generation",
        SCHEMA_TOOL,
        {"capability": "source_search"},
        allowed_server_ids={"provider-a"},
    )
    assert one["server"] == "provider-a"
    assert two["server"] == "provider-b"
    assert again == one
    # Different authorization scopes select independently. Repeating the same scope
    # must retain the already-reviewed owner and only live-revalidate its schema.
    assert selected == [frozenset({"provider-a"}), frozenset({"provider-b"})]
    assert revalidated == ["provider-a-lookup"]


def test_agent_runtime_propagates_exact_external_server_scope(tmp_path, monkeypatch) -> None:
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    ensure_schema_environment(runtime, "generation")
    runtime._schema_cache["generation"] = tuple(
        ExternalAgentBridge.tool_schemas("generation")
    )
    runtime._allowed_tool_cache["generation"] = frozenset(
        {CAPABILITIES_TOOL, SCHEMA_TOOL, CALL_TOOL}
    )
    seen: dict[str, object] = {}

    def fake_bridge(stage, name, payload, *, allowed_server_ids=None):
        seen["stage"] = stage
        seen["name"] = name
        seen["scope"] = frozenset(allowed_server_ids or ())
        return {"status": "PASS"}

    monkeypatch.setattr(runtime._external_bridge, "call", fake_bridge)
    result = runtime.call_scoped(
        "generation",
        CALL_TOOL,
        {"capability": "source_search", "arguments": {}},
        external_server_ids={"minecraft-dev", "mcmodding-docs"},
    )
    assert result["status"] == "PASS"
    assert seen == {
        "stage": "generation",
        "name": CALL_TOOL,
        "scope": frozenset({"minecraft-dev", "mcmodding-docs"}),
    }


class _Registry:
    def load_profile(self, name):
        return object()

    def role(self, profile, role):
        return SimpleNamespace(
            role=role,
            provider="local",
            adapter="llama_cpp",
            exclusive_gpu=False,
        )


class _ExternalRuntime:
    def __init__(self) -> None:
        self.scopes: list[frozenset[str]] = []
        self.unscoped_calls = 0

    def tool_schemas(self, stage):
        return ExternalAgentBridge.tool_schemas(stage)

    def call_scoped(self, stage, name, arguments, *, external_server_ids):
        self.scopes.append(frozenset(external_server_ids))
        return {"status": "PASS", "capability": arguments.get("capability")}

    def call(self, stage, name, arguments):
        self.unscoped_calls += 1
        raise AssertionError("external MCP model calls must never use the unscoped path")


class _ExternalAdapter:
    def __init__(self) -> None:
        self.count = 0

    def generate_turn(self, request):
        self.count += 1
        if self.count == 1:
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="mcp_1",
                        name=CALL_TOOL,
                        arguments={"capability": "source_search", "arguments": {}},
                        raw_arguments='{"capability":"source_search","arguments":{}}',
                    ),
                )
            )
        return GenerationResponse(content="done")


def test_model_router_enforces_minecraftcoder_server_scope(monkeypatch) -> None:
    runtime = _ExternalRuntime()
    adapter = _ExternalAdapter()
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)
    assert router.generate_text(
        "coder",
        ({"role": "user", "content": "inspect exact Minecraft API"},),
        tool_stage="generation",
    ) == "done"
    assert runtime.unscoped_calls == 0
    assert len(runtime.scopes) == 1
    scope = runtime.scopes[0]
    assert "minecraft-dev" in scope
    assert "mcmodding-docs" in scope
    assert "minecraft-wiki" in scope
    assert "fabric-game-runtime" not in scope
    assert "minecraft-player-agent" not in scope


def test_model_router_fails_closed_for_external_runtime_without_scope(monkeypatch) -> None:
    class UnscopedRuntime:
        def tool_schemas(self, stage):
            return ExternalAgentBridge.tool_schemas(stage)

        def call(self, stage, name, arguments):
            raise AssertionError("unscoped external path must not execute")

    adapter = _ExternalAdapter()
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: UnscopedRuntime(),
    )
    monkeypatch.setattr(router, "_new_text_adapter", lambda config, role: adapter)
    assert router.generate_text(
        "coder",
        ({"role": "user", "content": "inspect"},),
        tool_stage="generation",
    ) == "done"
    assert adapter.count == 2
