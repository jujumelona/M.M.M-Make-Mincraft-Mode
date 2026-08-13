from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Mapping, Sequence

from .agent_roles import (
    mcp_servers_for_model_role,
    routes_for_model_role,
    skills_for_model_role,
)
from .external_mcp_router import ExternalMCPRouter
from .skill_catalog import SkillContract, compile_skill_catalog


_EXTERNAL_AGENT_TOOLS = frozenset(
    {
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    }
)


@lru_cache(maxsize=8)
def _stage_contracts(stage: str) -> tuple[SkillContract, ...]:
    selected = stage.strip().lower()
    contracts = compile_skill_catalog()
    return tuple(
        contract
        for contract in contracts.values()
        if selected in contract.stages
    )


def _request_contracts(stage: str, model_role: str) -> tuple[SkillContract, ...]:
    stage_contracts = _stage_contracts(stage)
    assigned = skills_for_model_role(model_role)
    if not assigned:
        return stage_contracts
    return tuple(contract for contract in stage_contracts if contract.name in assigned)


def skills_for_tool(
    stage: str,
    tool: str,
    *,
    model_role: str = "",
) -> tuple[str, ...]:
    """Return canonical Skills allowed to route through ``tool`` for this role."""

    selected_tool = tool.strip()
    if not selected_tool or selected_tool in _EXTERNAL_AGENT_TOOLS:
        return ()
    return tuple(
        contract.name
        for contract in _request_contracts(stage, model_role)
        if selected_tool in contract.allowed_tools
    )


def build_agent_capability_context(
    stage: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    *,
    model_role: str = "",
) -> str:
    """Build compact, executable Skill/MCP guidance for the model tool chooser.

    The routing table in ``config/agent_roles.yaml`` is an execution contract, not
    documentation. Known model roles only see Skills assigned to their agent role(s)
    and external Minecraft MCP routes hosted by reviewed servers assigned to those
    roles. Unknown roles retain the stage-level fallback for backwards-compatible
    callers. Full Skill markdown and provider schemas stay out of the prompt; live
    schemas are requested on demand through the generic external MCP bridge.
    """

    selected = stage.strip().lower()
    exposed_tools = frozenset(_tool_names(tool_schemas))
    role_routes = routes_for_model_role(model_role)
    reviewed_servers = mcp_servers_for_model_role(model_role)

    skills: list[dict[str, Any]] = []
    for contract in _request_contracts(selected, model_role):
        model_tools = tuple(
            tool for tool in contract.allowed_tools if tool in exposed_tools
        )
        host_tools = tuple(
            tool for tool in contract.allowed_tools if tool not in exposed_tools
        )
        skills.append(
            {
                "name": contract.name,
                "description": contract.description,
                "model_tools": model_tools,
                "host_owned_tools": host_tools,
            }
        )

    external_capabilities: dict[str, tuple[str, ...]] = {}
    if exposed_tools & _EXTERNAL_AGENT_TOOLS:
        try:
            manifest = ExternalMCPRouter().capability_manifest(
                stage=selected,
                target=_environment_target(),
                max_access="read",
            )
            raw_capabilities = manifest.get("capabilities", {})
            if isinstance(raw_capabilities, Mapping):
                for name, raw_routes in raw_capabilities.items():
                    if not isinstance(raw_routes, list):
                        continue
                    servers = tuple(
                        sorted(
                            {
                                str(route.get("server", "")).strip()
                                for route in raw_routes
                                if isinstance(route, Mapping)
                                and str(route.get("server", "")).strip()
                                and (
                                    not role_routes
                                    or str(route.get("server", "")).strip()
                                    in reviewed_servers
                                )
                            }
                        )
                    )
                    if servers:
                        external_capabilities[str(name)] = servers
        except Exception:
            # Local first-party tools remain usable when an optional external MCP
            # registry cannot be loaded. Provider execution itself stays fail-closed.
            external_capabilities = {}

    payload = {
        "schema_version": "mmm/agent-capability-context-v2",
        "stage": selected,
        "model_role": model_role,
        "agent_roles": [route.name for route in role_routes],
        "reviewed_mcp_servers": sorted(reviewed_servers),
        "eligible_skills": skills,
        "external_minecraft_mcp_capabilities": external_capabilities,
        "routing_policy": (
            "Choose every relevant Skill route, not every route indiscriminately. "
            "Use model_tools directly. host_owned_tools belong to the durable host "
            "pipeline and must not be recreated recursively. For an external MCP "
            "capability, use external_mcp_schema when its live arguments are unknown, "
            "then external_mcp_call. The capability map lists reviewed provider servers "
            "available to this agent role. Prefer independent relevant evidence in "
            "parallel when it materially improves correctness; skip unrelated tools to "
            "avoid latency and token waste."
        ),
    }
    return "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _tool_names(tool_schemas: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    names: set[str] = set()
    for schema in tool_schemas:
        function = schema.get("function")
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", "")).strip()
        if name:
            names.add(name)
    return tuple(sorted(names))


def _environment_target() -> dict[str, str]:
    return {
        "minecraft_version": os.environ.get("MMM_MINECRAFT_VERSION", "").strip(),
        "loader": os.environ.get("MMM_LOADER", "fabric").strip() or "fabric",
        "mappings": os.environ.get("MMM_MAPPINGS", "").strip(),
    }
