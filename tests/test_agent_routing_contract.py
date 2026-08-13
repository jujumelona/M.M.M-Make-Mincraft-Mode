from __future__ import annotations

from minecraft_mod_ai.agent_roles import load_agent_role_routes
from minecraft_mod_ai.capability_plugins import PLUGIN_STATUSES
from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.skill_catalog import CANONICAL_SKILLS


_STALE_MCP_ALIASES = frozenset(
    {
        "minecraft-runtime-1201",
        "mineflayer-1201",
        "mmm-planning",
    }
)


def test_every_canonical_skill_is_assigned_to_an_agent_role() -> None:
    routes = load_agent_role_routes()
    assigned = {skill for route in routes for skill in route.skills}
    assert assigned == set(CANONICAL_SKILLS)


def test_agent_role_mcp_servers_all_exist_in_reviewed_registry() -> None:
    servers = set(ExternalMCPRegistry().servers)
    referenced = {
        server
        for route in load_agent_role_routes()
        for server in route.mcp_servers
    }
    assert referenced <= servers
    assert not (referenced & _STALE_MCP_ALIASES)


def test_plugin_required_mcp_servers_all_exist_in_reviewed_registry() -> None:
    servers = set(ExternalMCPRegistry().servers)
    referenced = {
        server
        for plugin in PLUGIN_STATUSES
        for server in plugin.required_mcp
    }
    assert referenced <= servers
    assert not (referenced & _STALE_MCP_ALIASES)


def test_agent_role_skill_names_are_unique_within_each_role() -> None:
    for route in load_agent_role_routes():
        assert len(route.skills) == len(set(route.skills))
        assert len(route.mcp_servers) == len(set(route.mcp_servers))
