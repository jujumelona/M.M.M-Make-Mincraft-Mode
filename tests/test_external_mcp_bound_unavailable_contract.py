from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import external_agent_bridge, external_mcp_router
from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.external_mcp_binding_contract import (
    _route_fingerprint,
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


def test_bound_provider_runtime_failure_is_unavailable_evidence(
    tmp_path: Path,
    monkeypatch,
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
    assert result["attempts"] == [
        {
            "server": "provider-a",
            "tool": "search",
            "status": "ERROR",
            "error": "RuntimeError: provider transport down",
        }
    ]
