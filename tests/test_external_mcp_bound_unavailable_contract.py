from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import external_agent_bridge, external_mcp_router
from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.external_mcp_binding_contract import (
    _route_fingerprint,
)
from minecraft_mod_ai.external_mcp_binding_contract import (
    install as install_binding,
)
from minecraft_mod_ai.external_mcp_router import ExternalMCPRouter


def _registry(tmp_path: Path) -> ExternalMCPRegistry:
    config = tmp_path / "external-mcp.yaml"
    config.write_text(
        """schema_version: mmm/external-mcp-registry-v2
servers:
  provider-a:
    status: enabled
    transport: stdio
    command: [fake-provider]
    version_policy: dynamic
    loaders: [fabric]
    trust: test
    capabilities:
      source_search:
        tool: search
        access: read
        stages: [generation]
        priority: 10
        target_args: {minecraft_version: version}
""",
        encoding="utf-8",
    )
    return ExternalMCPRegistry(config)


@pytest.mark.parametrize("grouped", [False, True])
def test_bound_provider_runtime_failure_is_unavailable_evidence(
    tmp_path: Path,
    monkeypatch,
    grouped,
) -> None:
    install_binding(external_agent_bridge, external_mcp_router)
    router = ExternalMCPRouter(_registry(tmp_path))
    route = router.registry.routes(
        "source_search",
        stage="generation",
        minecraft_version="26.2",
        loader="fabric",
        max_access="read",
    )[0]

    def fail_provider(*args, **kwargs):
        if grouped:
            try:
                from builtins import ExceptionGroup
            except ImportError:
                from exceptiongroup import ExceptionGroup
            raise ExceptionGroup("transport", [RuntimeError("provider transport down")])
        raise RuntimeError("provider transport down")

    monkeypatch.setattr(router, "_call_provider", fail_provider)

    result = router.invoke_bound(
        "source_search",
        stage="generation",
        server="provider-a",
        tool="search",
        expected_access="read",
        expected_route_sha256=_route_fingerprint(router, route),
        arguments={"query": "Item registry"},
        target={"minecraft_version": "26.2", "loader": "fabric", "mappings": "official"},
        max_access="read",
    )

    assert result["status"] == "UNAVAILABLE"
    assert result["evidence"] == []
    attempt = result["attempts"][0]
    assert len(result["attempts"]) == 1
    assert any(row["type"] == "RuntimeError" and row["message"] == "provider transport down"
               for row in attempt["exception_chain"])
    assert [{key: value for key, value in attempt.items() if key != "exception_chain"}] == [
        {
            "server": "provider-a",
            "tool": "search",
            "status": "ERROR",
            "error": ("ExceptionGroup: transport (1 sub-exception)" if grouped
                      else "RuntimeError: provider transport down"),
        }
    ]

def test_call_without_live_schema_returns_unavailable_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    install_binding(external_agent_bridge, external_mcp_router)
    bridge = external_agent_bridge.ExternalAgentBridge()
    router = ExternalMCPRouter(_registry(tmp_path))
    bridge._router = router

    async def unavailable_schema(*args, **kwargs):
        raise RuntimeError("schema provider down")

    monkeypatch.setattr(external_agent_bridge, "_provider_schema", unavailable_schema)

    result = bridge.call(
        "generation",
        external_agent_bridge.CALL_TOOL,
        {
            "capability": "source_search",
            "minecraft_version": "26.2",
            "loader": "fabric",
            "mappings": "official",
            "max_access": "read",
            "arguments": {"query": "Item registry"},
        },
    )

    assert result["status"] == "UNAVAILABLE"
    assert result["evidence"] == []
    assert result["attempts"]
    assert "schema provider down" in result["attempts"][0]["error"]
