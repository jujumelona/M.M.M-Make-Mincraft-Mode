from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Mapping, Sequence

from . import model_tool_aliases as _model_tool_aliases
from .agent_roles import (
    mcp_servers_for_model_role,
    routes_for_model_role,
    skills_for_model_role,
)
from .external_mcp_router import ExternalMCPRouter
from .skill_catalog import (
    REVIEWED_TOOL_STAGES,
    SkillContract,
    compile_skill_catalog,
)


_EXTERNAL_AGENT_TOOLS = frozenset(
    {
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    }
)


def _policy_model_role(stage: str, model_role: str) -> str:
    """Resolve the logical agent policy without changing model residency.

    Complete planning intentionally keeps one physical ``planner`` model session alive
    to avoid reloading the same local Qwen weights. Research turns inside that session
    are nevertheless owned by ResearchAgent, so only their Skill/MCP policy is switched
    to ``researcher`` when the host selects the research stage.
    """

    selected_stage = stage.strip().lower()
    selected_role = model_role.strip()
    if selected_stage == "research" and selected_role == "planner":
        return "researcher"
    return selected_role


def reviewed_mcp_servers_for_model_role(
    stage: str, model_role: str
) -> frozenset[str]:
    """Return reviewed external MCP servers for this logical agent turn."""
    return frozenset(
        mcp_servers_for_model_role(_policy_model_role(stage, model_role))
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


@lru_cache(maxsize=1)
def _manifest_router() -> ExternalMCPRouter:
    """Reuse the immutable reviewed provider registry on model-request hot paths."""

    return ExternalMCPRouter()


def _request_contracts(stage: str, model_role: str) -> tuple[SkillContract, ...]:
    stage_contracts = _stage_contracts(stage)
    assigned = skills_for_model_role(_policy_model_role(stage, model_role))
    if not assigned:
        return stage_contracts
    return tuple(contract for contract in stage_contracts if contract.name in assigned)


def _reviewed_stage_for_model_tool(name: str, stage: str) -> bool:
    canonical = _model_tool_aliases.canonical_model_tool(name)
    return stage in REVIEWED_TOOL_STAGES.get(canonical, frozenset())


def filter_tool_schemas_for_role(
    stage: str,
    model_role: str,
    tool_schemas: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Expose only tools reachable through reviewed stage/Skill/agent routes.

    Narrow model-facing ACIs inherit authorization from one canonical reviewed tool;
    aliases never create a second Skill/stage permission namespace. The physical model
    role may remain ``planner`` during a resident planning session; research-stage calls
    are evaluated against ResearchAgent through :func:`_policy_model_role`.
    """

    policy_role = _policy_model_role(stage, model_role)
    role_routes = routes_for_model_role(policy_role)
    if not role_routes:
        return tuple(tool_schemas)

    allowed = {
        tool
        for contract in _request_contracts(stage, policy_role)
        for tool in contract.allowed_tools
    }
    if mcp_servers_for_model_role(policy_role):
        allowed.update(_EXTERNAL_AGENT_TOOLS)

    selected_stage = stage.strip().lower()
    result: list[Mapping[str, Any]] = []
    for schema in tool_schemas:
        name = _schema_tool_name(schema)
        if not name:
            continue
        if name in _EXTERNAL_AGENT_TOOLS:
            if name in allowed:
                result.append(schema)
            continue
        canonical = _model_tool_aliases.canonical_model_tool(name)
        if canonical not in allowed:
            continue
        if not _reviewed_stage_for_model_tool(name, selected_stage):
            continue
        result.append(schema)
    return tuple(result)


def skills_for_tool(
    stage: str,
    tool: str,
    *,
    model_role: str = "",
) -> tuple[str, ...]:
    """Return canonical Skills allowed to route through ``tool`` for this agent."""

    selected_tool = tool.strip()
    selected_stage = stage.strip().lower()
    if not selected_tool or selected_tool in _EXTERNAL_AGENT_TOOLS:
        return ()
    canonical = _model_tool_aliases.canonical_model_tool(selected_tool)
    if selected_stage not in REVIEWED_TOOL_STAGES.get(canonical, frozenset()):
        return ()
    policy_role = _policy_model_role(selected_stage, model_role)
    return tuple(
        contract.name
        for contract in _request_contracts(selected_stage, policy_role)
        if canonical in contract.allowed_tools
    )


def build_agent_capability_context(
    stage: str,
    tool_schemas: Sequence[Mapping[str, Any]],
    *,
    model_role: str = "",
) -> str:
    """Build compact, executable Skill/MCP guidance for the model tool chooser.

    The routing table in ``config/agent_roles.yaml`` is an execution contract, not
    documentation. Known logical agent roles only see Skills assigned to their role and
    external Minecraft MCP routes hosted by reviewed servers assigned to those roles.
    A resident planner model may therefore execute a research turn while receiving only
    ResearchAgent policy, without loading a second copy of the same model.
    """

    selected = stage.strip().lower()
    policy_role = _policy_model_role(selected, model_role)
    exposed_tools = frozenset(_tool_names(tool_schemas))
    role_routes = routes_for_model_role(policy_role)
    reviewed_servers = reviewed_mcp_servers_for_model_role(selected, model_role)

    skills: list[dict[str, Any]] = []
    for contract in _request_contracts(selected, policy_role):
        stage_tools = tuple(
            tool
            for tool in contract.allowed_tools
            if selected in REVIEWED_TOOL_STAGES.get(tool, frozenset())
        )
        stage_tool_set = frozenset(stage_tools)
        model_tools = tuple(
            name
            for name in sorted(exposed_tools)
            if (
                name not in _EXTERNAL_AGENT_TOOLS
                and _model_tool_aliases.canonical_model_tool(name) in stage_tool_set
                and _reviewed_stage_for_model_tool(name, selected)
            )
        )
        represented_permissions = frozenset(
            _model_tool_aliases.canonical_model_tool(name) for name in model_tools
        )
        host_tools = tuple(
            tool for tool in stage_tools if tool not in represented_permissions
        )
        skills.append(
            {
                "name": contract.name,
                "description": contract.description,
                "activate_when": contract.activate_when,
                "model_tools": model_tools,
                "host_owned_tools": host_tools,
                "validators": contract.validators,
                "retry": contract.retry.to_dict(),
                "approvals": dict(contract.approvals),
                "forbidden_actions": contract.forbidden_actions,
                "exit": contract.exit.to_dict(),
            }
        )

    external_capabilities: dict[str, tuple[str, ...]] = {}
    external_access: dict[str, dict[str, str]] = {}
    if exposed_tools & _EXTERNAL_AGENT_TOOLS:
        try:
            manifest_max_access = "admin" if selected == "runtime" else "read"
            manifest = _manifest_router().capability_manifest(
                stage=selected,
                target=_environment_target(),
                max_access=manifest_max_access,
            )
            raw_capabilities = manifest.get("capabilities", {})
            if isinstance(raw_capabilities, Mapping):
                for name, raw_routes in raw_capabilities.items():
                    if not isinstance(raw_routes, list):
                        continue
                    selected_routes = tuple(
                        route
                        for route in raw_routes
                        if isinstance(route, Mapping)
                        and str(route.get("server", "")).strip()
                        and (
                            not role_routes
                            or str(route.get("server", "")).strip()
                            in reviewed_servers
                        )
                    )
                    servers = tuple(
                        sorted(
                            {
                                str(route.get("server", "")).strip()
                                for route in selected_routes
                            }
                        )
                    )
                    if not servers:
                        continue
                    capability = str(name)
                    external_capabilities[capability] = servers
                    external_access[capability] = {
                        str(route.get("server", "")).strip(): str(
                            route.get("access", "read")
                        ).strip()
                        or "read"
                        for route in selected_routes
                    }
        except Exception:
            external_capabilities = {}
            external_access = {}

    payload = {
        "schema_version": "mmm/agent-capability-context-v4",
        "stage": selected,
        "model_role": policy_role,
        "execution_model_role": model_role,
        "agent_roles": [route.name for route in role_routes],
        "reviewed_mcp_servers": sorted(reviewed_servers),
        "eligible_skills": skills,
        "external_minecraft_mcp_capabilities": external_capabilities,
        "external_minecraft_mcp_access": external_access,
        "routing_policy": (
            "Choose every relevant Skill route, not every route indiscriminately. "
            "The execution model may stay resident under a different physical role; "
            "model_role and agent_roles in this context are the authoritative logical "
            "policy for this stage. Obey the selected Skill's validators, approvals, "
            "forbidden_actions, retry and exit contract. Use model_tools directly. "
            "host_owned_tools belong to the durable host pipeline and must not be "
            "recreated recursively. For an external MCP capability, use "
            "external_mcp_schema when its live arguments are unknown, then "
            "external_mcp_call. The capability and access maps list reviewed provider "
            "routes available to this agent role. Outside runtime, external MCP access "
            "is read-only. Runtime write/admin capabilities are discoverable only so "
            "disposable playtests can use them and must be called with "
            "disposable_runtime=true; the execution router remains fail-closed. During "
            "production, use an adaptive evidence loop: retrieve fresh project or "
            "exact-version API evidence, inspect retrieval coverage/relevance, change "
            "the query or reviewed source when evidence is weak, generate or repair, "
            "then treat compiler/JDT/runtime feedback as a new observation and retrieve "
            "again when it introduces uncertainty. Never guess exact Minecraft/Fabric/"
            "mapping/dependency/Java API facts from model memory when reviewed evidence "
            "can resolve them. Prefer independent read-only evidence in parallel when it "
            "materially improves correctness; keep state changes ordered and skip "
            "unrelated tools. Preserve host safety invariants: disposable_runtime=true; "
            "retrieved_context_can_authorize=false; writes_require_approval_hash=true."
        ),
    }
    return "MMM reviewed Skill/tool/Minecraft-MCP routing context:\n" + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _schema_tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _tool_names(tool_schemas: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                name
                for schema in tool_schemas
                if (name := _schema_tool_name(schema))
            }
        )
    )


def _environment_target() -> dict[str, str]:
    return {
        "minecraft_version": os.environ.get("MMM_MINECRAFT_VERSION", "").strip(),
        "loader": os.environ.get("MMM_LOADER", "fabric").strip() or "fabric",
        "mappings": os.environ.get("MMM_MAPPINGS", "").strip(),
    }
