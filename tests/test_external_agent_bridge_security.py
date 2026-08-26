from __future__ import annotations

import pytest

import minecraft_mod_ai.external_agent_bridge as external_agent_bridge_module
from minecraft_mod_ai.external_agent_bridge import (
    CALL_TOOL,
    SCHEMA_TOOL,
    ExternalAgentBridge,
    ExternalAgentBridgeError,
)

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "additionalProperties": False,
}


class _Registry:
    def routes(self, capability: str, **kwargs: object) -> list[dict[str, object]]:
        assert capability == "docs.lookup"
        access = str(kwargs.get("max_access", "read"))
        return [
            {
                "server": "reviewed-docs",
                "entry": {
                    "status": "enabled",
                    "transport": "stdio",
                    "command": ["reviewed-docs"],
                    "trust": "test",
                    "fixture_access": access,
                },
                "route": {
                    "tool": "lookup",
                    "access": access,
                    "target_args": {
                        "minecraft_version": "mc_version",
                        "loader": "loader_name",
                    },
                },
            }
        ]


class _Router:
    def __init__(self) -> None:
        self.registry = _Registry()
        self.last_invoke: dict[str, object] | None = None

    @staticmethod
    def _configured(entry: dict[str, object]) -> bool:
        return entry.get("status") == "enabled"

    @staticmethod
    def _child_env(entry: dict[str, object]) -> dict[str, str]:
        return {}

    @staticmethod
    def _server_url(entry: dict[str, object]) -> str:
        return ""

    def invoke_bound(self, capability: str, **kwargs: object) -> dict[str, object]:
        self.last_invoke = {"capability": capability, **kwargs}
        return {"status": "PASS"}


def _bridge() -> tuple[ExternalAgentBridge, _Router]:
    bridge = ExternalAgentBridge()
    router = _Router()
    bridge._router = router
    return bridge, router


def _install_fake_provider_schema(monkeypatch, calls: list[str] | None = None) -> None:
    async def fake_provider_schema(entry, *, tool, env, url, timeout_seconds):
        assert tool == "lookup"
        if calls is not None:
            calls.append(str(entry["fixture_access"]))
        return {
            "description": "fixture lookup",
            "input_schema": dict(_INPUT_SCHEMA),
        }

    monkeypatch.setattr(
        external_agent_bridge_module,
        "_provider_schema",
        fake_provider_schema,
    )


def test_external_mcp_rejects_truthy_non_boolean_disposable_flag() -> None:
    bridge, router = _bridge()

    with pytest.raises(ExternalAgentBridgeError, match="must be a boolean"):
        bridge.call(
            "runtime",
            CALL_TOOL,
            {
                "capability": "docs.lookup",
                "arguments": {},
                "max_access": "write",
                "disposable_runtime": "false",
            },
        )

    assert router.last_invoke is None


def test_external_mcp_strips_model_owned_platform_target_overrides(monkeypatch) -> None:
    bridge, router = _bridge()
    _install_fake_provider_schema(monkeypatch)

    bridge.call(
        "research",
        CALL_TOOL,
        {
            "capability": "docs.lookup",
            "minecraft_version": "mmm-host-target",
            "loader": "fabric",
            "mappings": "host-selected-mappings",
            "arguments": {
                "query": "registry api",
                "mc_version": "mmm-model-override",
                "loader_name": "forge",
            },
        },
        allowed_server_ids={"reviewed-docs"},
    )

    assert router.last_invoke is not None
    assert router.last_invoke["target"] == {
        "minecraft_version": "mmm-host-target",
        "loader": "fabric",
        "mappings": "host-selected-mappings",
    }
    assert router.last_invoke["arguments"] == {"query": "registry api"}
    assert router.last_invoke["allowed_server_ids"] == frozenset({"reviewed-docs"})


def test_external_mcp_schema_bindings_are_partitioned_and_live_refreshed_by_access(
    monkeypatch,
) -> None:
    bridge, _ = _bridge()
    calls: list[str] = []
    _install_fake_provider_schema(monkeypatch, calls)
    common = {
        "capability": "docs.lookup",
        "minecraft_version": "mmm-host-target",
        "loader": "fabric",
    }

    read_schema = bridge.call(
        "runtime",
        SCHEMA_TOOL,
        {**common, "max_access": "read"},
    )
    write_schema = bridge.call(
        "runtime",
        SCHEMA_TOOL,
        {**common, "max_access": "write"},
    )
    read_again = bridge.call(
        "runtime",
        SCHEMA_TOOL,
        {**common, "max_access": "read"},
    )

    assert read_schema["access"] == "read"
    assert write_schema["access"] == "write"
    assert read_again["access"] == "read"
    # Explicit schema discovery is a live trust boundary. Re-querying the same
    # scope must not resurrect the old indefinite provider-blind cache behavior.
    assert calls == ["read", "write", "read"]
