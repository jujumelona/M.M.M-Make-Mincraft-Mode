from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.external_mcp_binding_contract import install as install_binding
from minecraft_mod_ai.mcp_schema_integrity_contract import (
    MCPSchemaIntegrityError,
    validate_raw_tool_rows,
)


def _object_schema(**properties: dict) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }


def test_raw_tools_list_rejects_duplicate_names() -> None:
    rows = [
        {"name": "same", "description": "a", "input_schema": _object_schema()},
        {"name": "same", "description": "b", "input_schema": _object_schema()},
    ]
    with pytest.raises(MCPSchemaIntegrityError, match="duplicate MCP tool name 'same'"):
        validate_raw_tool_rows(rows, surface="first-party:generation")


def test_raw_tools_list_rejects_malformed_input_schema() -> None:
    rows = [{"name": "broken", "description": "", "input_schema": ["not", "object"]}]
    with pytest.raises(MCPSchemaIntegrityError, match="input schema must be an object"):
        validate_raw_tool_rows(rows, surface="first-party:generation")


def test_raw_tools_list_rejects_non_object_root_schema() -> None:
    rows = [
        {
            "name": "broken",
            "description": "",
            "input_schema": {"type": "array", "items": {"type": "string"}},
        }
    ]
    with pytest.raises(MCPSchemaIntegrityError, match="root type must be 'object'"):
        validate_raw_tool_rows(rows, surface="external-provider")


def _fake_modules():
    schema_a = _object_schema(query={"type": "string"})
    schema_b = _object_schema(symbol={"type": "string"})
    live_schemas = {
        "provider-a": schema_a,
        "provider-b": schema_b,
    }
    routes = [
        {
            "server": "provider-a",
            "entry": {"status": "enabled", "provider_key": "provider-a"},
            "route": {"tool": "search_a", "access": "read", "target_args": {}},
        },
        {
            "server": "provider-b",
            "entry": {"status": "enabled", "provider_key": "provider-b"},
            "route": {"tool": "search_b", "access": "read", "target_args": {}},
        },
    ]

    class Registry:
        def routes(self, capability, **kwargs):
            assert capability == "source_search"
            return list(routes)

    class RouteTarget:
        def __init__(self, value):
            value = value or {}
            self.minecraft_version = str(value.get("minecraft_version", ""))
            self.loader = str(value.get("loader", "fabric") or "fabric")
            self.mappings = str(value.get("mappings", ""))
            self.mapping = "yarn"

        @classmethod
        def from_value(cls, value):
            return cls(value)

        def to_dict(self):
            return {
                "minecraft_version": self.minecraft_version,
                "loader": self.loader,
                "mappings": self.mappings,
                "mapping": self.mapping,
            }

    class FakeRouter:
        def __init__(self):
            self.registry = Registry()
            self.calls: list[tuple[str, str, dict]] = []
            self.live_schemas = live_schemas

        @staticmethod
        def _configured(entry):
            return entry.get("status") == "enabled"

        @staticmethod
        def _child_env(entry):
            return {}

        @staticmethod
        def _server_url(entry):
            return ""

        @staticmethod
        def _arguments_for_route(arguments, route, target):
            return dict(arguments)

        def _call_provider(self, server, entry, *, tool, arguments):
            self.calls.append((server, tool, dict(arguments)))
            return {"server_info": {"name": server}, "result": {"server": server}}

        @staticmethod
        def _validate_reported_target(result, route, target):
            return None

    router = FakeRouter()

    def target(payload):
        return {
            "minecraft_version": str(payload.get("minecraft_version", "")),
            "loader": str(payload.get("loader", "fabric") or "fabric"),
            "mappings": str(payload.get("mappings", "")),
        }

    class BridgeError(RuntimeError):
        pass

    class FakeBridge:
        def __init__(self):
            self.timeout_seconds = 10.0
            self._lock = threading.RLock()
            self._schema_cache = {}
            self._router = router
            self.schema_queries = 0

        @staticmethod
        def tool_schemas(stage):
            if stage != "generation":
                return ()
            return (
                {
                    "type": "function",
                    "function": {
                        "name": "external_mcp_call",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "capability": {"type": "string"},
                                "arguments": {"type": "object"},
                                "corroborate": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 4,
                                    "default": 1,
                                },
                            },
                        },
                    },
                },
            )

        def call(self, stage, name, payload, *, allowed_server_ids=None):
            if name != "external_mcp_schema":
                raise AssertionError("binding contract must own external_mcp_call")
            self.schema_queries += 1
            route = self._router.registry.routes("source_search")[0]
            server = route["server"]
            return {
                "schema_version": "test",
                "capability": "source_search",
                "stage": stage,
                "target": target(payload),
                "server": server,
                "tool": route["route"]["tool"],
                "access": route["route"].get("access", "read"),
                "trust": "reviewed",
                "target_args_injected_by_router": dict(route["route"].get("target_args", {})),
                "description": "",
                "input_schema": dict(self._router.live_schemas[server]),
                "status": "PASS",
            }

        def _external_router(self):
            return self._router

        def _run_async(self, function, *args):
            return asyncio.run(function(*args))

        @staticmethod
        def _reserved_target_arguments(router, **kwargs):
            return frozenset()

    async def provider_schema(entry, *, tool, env, url, timeout_seconds):
        provider_key = entry["provider_key"]
        return {
            "description": "",
            "input_schema": dict(router.live_schemas[provider_key]),
        }

    bridge_module = SimpleNamespace(
        ExternalAgentBridge=FakeBridge,
        ExternalAgentBridgeError=BridgeError,
        CAPABILITIES_TOOL="external_mcp_capabilities",
        SCHEMA_TOOL="external_mcp_schema",
        CALL_TOOL="external_mcp_call",
        AGENT_STAGES=frozenset({"generation", "runtime"}),
        _target=target,
        _provider_schema=provider_schema,
    )
    router_module = SimpleNamespace(
        ExternalMCPRouter=FakeRouter,
        MCPRouteTarget=RouteTarget,
        ExternalMCPError=RuntimeError,
        _sha256=lambda value: hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    install_binding(bridge_module, router_module)
    return bridge_module, router, routes


def test_external_call_schema_advertises_single_bound_provider() -> None:
    bridge_module, _, _ = _fake_modules()
    rows = bridge_module.ExternalAgentBridge.tool_schemas("generation")
    corroborate = rows[0]["function"]["parameters"]["properties"]["corroborate"]
    assert corroborate["maximum"] == 1
    assert corroborate["default"] == 1


def test_schema_provider_remains_execution_provider_after_route_reorder() -> None:
    bridge_module, router, routes = _fake_modules()
    bridge = bridge_module.ExternalAgentBridge()
    schema = bridge.call(
        "generation",
        "external_mcp_schema",
        {"capability": "source_search"},
    )
    assert schema["server"] == "provider-a"

    routes.reverse()
    result = bridge.call(
        "generation",
        "external_mcp_call",
        {"capability": "source_search", "arguments": {"query": "BlockEntity"}},
    )

    assert result["status"] == "PASS"
    assert router.calls == [
        ("provider-a", "search_a", {"query": "BlockEntity"})
    ]


def test_schema_drift_invalidates_binding_before_provider_execution() -> None:
    bridge_module, router, _ = _fake_modules()
    bridge = bridge_module.ExternalAgentBridge()
    bridge.call(
        "generation",
        "external_mcp_schema",
        {"capability": "source_search"},
    )
    router.live_schemas["provider-a"] = _object_schema(path={"type": "string"})

    with pytest.raises(
        bridge_module.ExternalAgentBridgeError,
        match="schema changed after discovery",
    ):
        bridge.call(
            "generation",
            "external_mcp_call",
            {"capability": "source_search", "arguments": {"query": "x"}},
        )
    assert router.calls == []


def test_route_access_drift_invalidates_binding_before_provider_execution() -> None:
    bridge_module, router, routes = _fake_modules()
    bridge = bridge_module.ExternalAgentBridge()
    bridge.call(
        "generation",
        "external_mcp_schema",
        {"capability": "source_search"},
    )
    routes[0]["route"]["access"] = "admin"

    with pytest.raises(
        bridge_module.ExternalAgentBridgeError,
        match="access changed after schema discovery",
    ):
        bridge.call(
            "generation",
            "external_mcp_call",
            {"capability": "source_search", "arguments": {"query": "x"}},
        )
    assert router.calls == []


def test_provider_configuration_drift_invalidates_binding_before_execution() -> None:
    bridge_module, router, routes = _fake_modules()
    bridge = bridge_module.ExternalAgentBridge()
    bridge.call(
        "generation",
        "external_mcp_schema",
        {"capability": "source_search"},
    )
    routes[0]["entry"]["command"] = ["different-provider-process"]

    with pytest.raises(
        bridge_module.ExternalAgentBridgeError,
        match="provider/route identity changed after schema discovery",
    ):
        bridge.call(
            "generation",
            "external_mcp_call",
            {"capability": "source_search", "arguments": {"query": "x"}},
        )
    assert router.calls == []


def test_call_without_prior_schema_establishes_fresh_binding() -> None:
    bridge_module, router, _ = _fake_modules()
    bridge = bridge_module.ExternalAgentBridge()
    result = bridge.call(
        "generation",
        "external_mcp_call",
        {"capability": "source_search", "arguments": {"query": "fresh"}},
    )
    assert result["status"] == "PASS"
    assert bridge.schema_queries == 1
    assert router.calls[0][0:2] == ("provider-a", "search_a")
