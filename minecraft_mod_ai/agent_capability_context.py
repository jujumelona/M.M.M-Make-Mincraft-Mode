from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Mapping, Sequence

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


def skills_for_tool(stage: str, tool: str) -> tuple[str, ...]:
    """Return canonical Skills whose reviewed policy routes through ``tool``."""

    selected_tool = tool.strip()
    if not selected_tool or selected_tool in _EXTERNAL_AGENT_TOOLS:
        return ()
    return tuple(
        contract.name
        for contract in _stage_contracts(stage)
        if selected_tool in contract.allowed_tools
    )


def build_agent_capability_context(
    stage: str,
    tool_schemas: Sequence[Mapping[str, Any]],
) -> str:
    """Build compact stage-local Skill/MCP guidance for the model tool chooser.

    Skill markdown and plugin manifests are intentionally not injected wholesale.
    The model sees only Skills eligible in the current stage, which of their tools
    are actually model-callable in this turn, and the reviewed external Minecraft
    MCP capability names available for the active target. This keeps all reviewed
    routes reachable without paying the token and latency cost of advertising every
    repository capability on every model call.
    """

    selected = stage.strip().lower()
    exposed_tools = frozenset(_tool_names(tool_schemas))
    skills: list[dict[str, Any]] = []
    for contract in _stage_contracts(selected):
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

    external_capabilities: tuple[str, ...] = ()
    if exposed_tools & _EXTERNAL_AGENT_TOOLS:
        try:
            manifest = ExternalMCPRouter().capability_manifest(
                stage=selected,
                target=_environment_target(),
                max_access="read",
            )
            raw_capabilities = manifest.get("capabilities", {})
            if isinstance(raw_capabilities, Mapping):
                external_capabilities = tuple(sorted(str(name) for name in raw_capabilities))
        except Exception:
            # Local first-party tools must remain usable if an optional external
            # registry is unavailable or malformed. Provider execution itself is
            # still fail-closed in ExternalMCPRouter.
            external_capabilities = ()

    payload = {
        "schema_version": "mmm/agent-capability-context-v1",
        "stage": selected,
        "eligible_skills": skills,
        "external_minecraft_mcp_capabilities": external_capabilities,
        "routing_policy": (
            "Choose every relevant Skill route, not every route indiscriminately. "
            "Use model_tools directly. host_owned_tools belong to the durable host "
            "pipeline and must not be recreated recursively. For an external MCP "
            "capability, use external_mcp_schema when its live arguments are unknown, "
            "then external_mcp_call. Prefer independent relevant evidence in parallel "
            "when it materially improves correctness; skip unrelated tools to avoid "
            "latency and token waste."
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
