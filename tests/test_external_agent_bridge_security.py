from __future__ import annotations

import pytest

from minecraft_mod_ai.external_agent_bridge import (
    CALL_TOOL,
    ExternalAgentBridge,
    ExternalAgentBridgeError,
)


class _Registry:
    def routes(self, capability: str, **_: object) -> list[dict[str, object]]:
        assert capability == "docs.lookup"
        return [
            {
                "server": "reviewed-docs",
                "route": {
                    "tool": "lookup",
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

    def invoke(self, capability: str, **kwargs: object) -> dict[str, object]:
        self.last_invoke = {"capability": capability, **kwargs}
        return {"status": "PASS"}


def _bridge() -> tuple[ExternalAgentBridge, _Router]:
    bridge = ExternalAgentBridge()
    router = _Router()
    bridge._router = router
    return bridge, router


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


def test_external_mcp_strips_model_owned_platform_target_overrides() -> None:
    bridge, router = _bridge()

    bridge.call(
        "research",
        CALL_TOOL,
        {
            "capability": "docs.lookup",
            "minecraft_version": "1.20.1",
            "loader": "fabric",
            "arguments": {
                "query": "registry api",
                "mc_version": "1.12.2",
                "loader_name": "forge",
            },
        },
        allowed_server_ids={"reviewed-docs"},
    )

    assert router.last_invoke is not None
    assert router.last_invoke["target"] == {
        "minecraft_version": "1.20.1",
        "loader": "fabric",
        "mappings": "yarn-1.20.1+build.1",
    }
    assert router.last_invoke["arguments"] == {"query": "registry api"}
    assert router.last_invoke["allowed_server_ids"] == frozenset({"reviewed-docs"})
