from __future__ import annotations

from pathlib import Path

import pytest

from minecraft_mod_ai import agent_roles


def _route_yaml(name: str, server: str) -> str:
    return f"""schema_version: mmm/agent-roles-v3
agents:
  {name}:
    model_role: coder
    skills: [minecraft_mod_authoring]
    mcp_servers: [{server}]
"""


def test_duplicate_agent_owner_is_rejected_before_yaml_last_writer_wins() -> None:
    text = """schema_version: mmm/agent-roles-v3
agents:
  MinecraftCoder:
    model_role: coder
    skills: [safe]
    mcp_servers: [safe-provider]
  MinecraftCoder:
    model_role: coder
    skills: [unsafe]
    mcp_servers: [other-provider]
"""

    with pytest.raises(
        ValueError,
        match="Duplicate agent role routing contract YAML key 'MinecraftCoder'",
    ):
        agent_roles._parse_agent_role_routes(text)


def test_duplicate_permission_field_is_rejected_before_override() -> None:
    text = """schema_version: mmm/agent-roles-v3
agents:
  MinecraftCoder:
    model_role: coder
    skills: [minecraft_mod_authoring]
    mcp_servers: [safe-provider]
    mcp_servers: [other-provider]
"""

    with pytest.raises(
        ValueError,
        match="Duplicate agent role routing contract YAML key 'mcp_servers'",
    ):
        agent_roles._parse_agent_role_routes(text)


def test_long_lived_loader_observes_same_path_permission_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "agent_roles.yaml"
    path.write_text(_route_yaml("MinecraftCoder", "provider-a"), encoding="utf-8")
    monkeypatch.setattr(agent_roles, "config_path", lambda _: path)

    first = agent_roles.load_agent_role_routes()
    assert first[0].mcp_servers == ("provider-a",)

    path.write_text(_route_yaml("MinecraftCoder", "provider-b"), encoding="utf-8")
    second = agent_roles.load_agent_role_routes()
    assert second[0].mcp_servers == ("provider-b",)
