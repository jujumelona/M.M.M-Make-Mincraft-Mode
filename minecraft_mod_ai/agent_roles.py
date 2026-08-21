from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .config_paths import config_path
from .strict_yaml import safe_load_unique_keys


@dataclass(frozen=True)
class AgentRoleRoute:
    name: str
    model_roles: tuple[str, ...]
    skills: tuple[str, ...]
    mcp_servers: tuple[str, ...]


def load_agent_role_routes() -> tuple[AgentRoleRoute, ...]:
    """Load the reviewed model-role -> Skill/MCP routing contract.

    The file is tiny, so read its current bytes on every lookup and cache only parsing
    by exact content. This keeps permission/routing changes visible in a long-lived
    process while still avoiding repeated YAML construction for unchanged content.
    """

    path = config_path("agent_roles.yaml")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("MMM agent role routing contract must be UTF-8.") from exc
    return _parse_agent_role_routes(text)


@lru_cache(maxsize=8)
def _parse_agent_role_routes(text: str) -> tuple[AgentRoleRoute, ...]:
    raw = safe_load_unique_keys(text, source="agent role routing contract")
    if not isinstance(raw, dict) or raw.get("schema_version") != "mmm/agent-roles-v3":
        raise ValueError("Unsupported MMM agent role routing contract.")
    agents = raw.get("agents")
    if not isinstance(agents, dict) or not agents:
        raise ValueError("MMM agent role routing contract contains no agents.")

    routes: list[AgentRoleRoute] = []
    for name, entry in agents.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
            raise ValueError(f"Invalid MMM agent role entry: {name!r}")
        model_roles = _model_roles(entry)
        skills = _string_tuple(entry.get("skills"), field=f"{name}.skills")
        mcp_servers = _string_tuple(
            entry.get("mcp_servers"), field=f"{name}.mcp_servers"
        )
        if not model_roles:
            raise ValueError(f"MMM agent role {name!r} has no model role binding.")
        if not skills:
            raise ValueError(f"MMM agent role {name!r} has no Skill binding.")
        routes.append(
            AgentRoleRoute(
                name=name.strip(),
                model_roles=model_roles,
                skills=skills,
                mcp_servers=mcp_servers,
            )
        )
    return tuple(routes)


def routes_for_model_role(model_role: str) -> tuple[AgentRoleRoute, ...]:
    selected = model_role.strip()
    if not selected:
        return ()
    return tuple(
        route for route in load_agent_role_routes() if selected in route.model_roles
    )


def skills_for_model_role(model_role: str) -> frozenset[str]:
    return frozenset(
        skill
        for route in routes_for_model_role(model_role)
        for skill in route.skills
    )


def mcp_servers_for_model_role(model_role: str) -> frozenset[str]:
    return frozenset(
        server
        for route in routes_for_model_role(model_role)
        for server in route.mcp_servers
    )


def _model_roles(entry: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in ("model_role", "safe_model_role"):
        value = str(entry.get(key, "")).strip()
        if value and value not in values:
            values.append(value)
    return tuple(values)


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"MMM agent role field {field!r} must be a string list.")
    result: list[str] = []
    for item in value:
        selected = item.strip()
        if selected not in result:
            result.append(selected)
    return tuple(result)