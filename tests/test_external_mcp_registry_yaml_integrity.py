from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai.external_mcp import ExternalMCPRegistry


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "external-mcp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_duplicate_server_key_is_rejected_before_yaml_last_writer_wins(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: mmm/external-mcp-registry-v2
servers:
  docs:
    status: enabled
    transport: stdio
    command: [first]
    version_policy: dynamic
    capabilities: {}
  docs:
    status: enabled
    transport: stdio
    command: [second]
    version_policy: dynamic
    capabilities: {}
""",
    )

    with pytest.raises(ValueError, match="Duplicate external MCP registry YAML key 'docs'"):
        ExternalMCPRegistry(path)


def test_duplicate_capability_key_is_rejected_before_route_override(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: mmm/external-mcp-registry-v2
servers:
  docs:
    status: enabled
    transport: stdio
    command: [docs]
    version_policy: dynamic
    capabilities:
      source_search:
        tool: safe_lookup
        access: read
      source_search:
        tool: unsafe_mutate
        access: write
""",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate external MCP registry YAML key 'source_search'",
    ):
        ExternalMCPRegistry(path)


def test_yaml_merge_precedence_is_forbidden_in_reviewed_registry(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        """schema_version: mmm/external-mcp-registry-v2
base: &provider
  status: enabled
  transport: stdio
  command: [docs]
  version_policy: dynamic
  capabilities: {}
servers:
  docs:
    <<: *provider
""",
    )

    with pytest.raises(ValueError, match="YAML merge keys are forbidden"):
        ExternalMCPRegistry(path)
