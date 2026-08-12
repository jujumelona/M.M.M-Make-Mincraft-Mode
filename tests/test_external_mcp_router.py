from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.external_mcp_router import (
    ExternalMCPError,
    ExternalMCPRouter,
    MCPRouteTarget,
)


def _registry(tmp_path: Path) -> ExternalMCPRegistry:
    path = tmp_path / "external-mcp.yaml"
    path.write_text(
        """schema_version: mmm/external-mcp-registry-v2
servers:
  dynamic-primary:
    status: enabled
    transport: stdio
    command: [fake-primary]
    version_policy: provider_reported
    loaders: [fabric]
    trust: test
    capabilities:
      source_search:
        tool: search
        access: read
        stages: [research, quality]
        priority: 10
        target_args: {minecraft_version: version, mapping: mapping}
      dangerous:
        tool: mutate
        access: admin
        stages: [runtime]
        priority: 10
  dynamic-fallback:
    status: enabled
    transport: stdio
    command: [fake-fallback]
    version_policy: dynamic
    loaders: [fabric]
    trust: test
    capabilities:
      source_search:
        tool: search2
        access: read
        stages: [research]
        priority: 20
  exact-old:
    status: enabled
    transport: stdio
    command: [fake-exact]
    version_policy: exact
    target_versions: [1.20.1]
    loaders: [fabric]
    trust: test
    capabilities:
      source_search:
        tool: old_search
        access: read
        stages: [research]
        priority: 1
""",
        encoding="utf-8",
    )
    return ExternalMCPRegistry(path)


def test_future_version_routes_without_static_allowlist(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    rows = registry.routes(
        "source_search",
        stage="research",
        minecraft_version="27.0",
        loader="fabric",
        max_access="read",
    )
    assert [row["server"] for row in rows] == [
        "dynamic-primary",
        "dynamic-fallback",
    ]


def test_exact_provider_is_only_used_for_declared_target(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    old = registry.routes(
        "source_search",
        stage="research",
        minecraft_version="1.20.1",
        loader="fabric",
    )
    future = registry.routes(
        "source_search",
        stage="research",
        minecraft_version="27.0",
        loader="fabric",
    )
    assert [row["server"] for row in old][0] == "exact-old"
    assert "exact-old" not in [row["server"] for row in future]


def test_non_runtime_federation_is_read_only(tmp_path: Path) -> None:
    router = ExternalMCPRouter(_registry(tmp_path))
    with pytest.raises(ExternalMCPError, match="read-only"):
        router.invoke(
            "source_search",
            stage="research",
            target={"minecraft_version": "27.0", "loader": "fabric"},
            max_access="admin",
        )


def test_runtime_admin_requires_disposable_instance(tmp_path: Path) -> None:
    router = ExternalMCPRouter(_registry(tmp_path))
    with pytest.raises(ExternalMCPError, match="disposable runtime"):
        router.invoke(
            "dangerous",
            stage="runtime",
            target={"minecraft_version": "27.0", "loader": "fabric"},
            max_access="admin",
            disposable_runtime=False,
        )


def test_priority_fallback_and_target_argument_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = ExternalMCPRouter(_registry(tmp_path))
    calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_call(server_name, entry, *, tool, arguments):
        calls.append((server_name, tool, dict(arguments)))
        if server_name == "dynamic-primary":
            raise RuntimeError("primary unavailable")
        return {
            "server_info": {"name": server_name},
            "result": {
                "structured": {
                    "minecraft_version": "27.0",
                    "hits": ["ok"],
                },
                "parsed_text": None,
                "text": [],
                "other_content": [],
            },
        }

    monkeypatch.setattr(router, "_call_provider", fake_call)
    bundle = router.invoke(
        "source_search",
        stage="research",
        arguments={"query": "Entity"},
        target={
            "minecraft_version": "27.0",
            "loader": "fabric",
            "mappings": "mojang",
        },
    )

    assert bundle["status"] == "PASS"
    assert [call[0] for call in calls] == ["dynamic-primary", "dynamic-fallback"]
    first_args = calls[0][2]
    assert first_args["version"] == "27.0"
    assert first_args["mapping"] == "mojmap"
    assert bundle["evidence"][0]["server"] == "dynamic-fallback"


def test_explicit_provider_target_conflict_is_rejected() -> None:
    target = MCPRouteTarget.from_value(
        {"minecraft_version": "27.0", "loader": "fabric", "mappings": "mojang"}
    )
    with pytest.raises(ExternalMCPError, match="conflicts with the approved PlatformLock"):
        ExternalMCPRouter._validate_reported_target(
            {
                "structured": {
                    "target": {"minecraft_version": "1.20.1"},
                }
            },
            {},
            target,
        )


def test_checked_in_registry_is_valid_and_capability_routed() -> None:
    registry = ExternalMCPRegistry()
    future = registry.routes(
        "official_mod_docs",
        stage="research",
        minecraft_version="27.0",
        loader="fabric",
    )
    assert any(row["server"] == "mcmodding-docs" for row in future)
    assert registry.server("minecraft-dev")["version_policy"] == "provider_reported"
    assert registry.server("fabric-game-runtime")["default_url"].endswith("8765/mcp")
    assert registry.server("fabric-game-client-runtime")["default_url"].endswith("8766/mcp")
