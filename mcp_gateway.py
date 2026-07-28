"""Compatibility facade backed by the real MMM MCP tool service.

This module no longer reports unconditional success. It accepts only the closed tool
set implemented by :class:`minecraft_mod_ai.mcp_tools.MMMToolService` and returns the
actual result or a failed envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minecraft_mod_ai.mcp_tools import MMMToolService


@dataclass(frozen=True)
class AuthContext:
    principal: str = "agent:coder"
    role: str = "implementer"


@dataclass(frozen=True)
class ExecutionLimits:
    timeout_s: int = 600
    network_policy: str = "deny"


@dataclass(frozen=True)
class MCPRequestEnvelope:
    project_id: str
    plan_version: int
    artifact_revision: str
    request_id: str
    auth_context: AuthContext
    limits: ExecutionLimits
    tool_name: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MCPResponseEnvelope:
    status: str
    result: dict[str, Any] | None = None
    error: str | None = None


class DomainMCPServerRegistry:
    _ALLOWED = frozenset(
        {
            "plan_game",
            "revise_plan",
            "approve_plan",
            "search_project_rag",
            "inspect_existing_mod",
            "generate_fabric_project",
            "generate_assets",
            "generate_world_ir",
            "run_static_validation",
            "run_gradle_build",
            "run_gametest",
            "inspect_jar",
            "package_release",
        }
    )

    def __init__(self, service: MMMToolService | None = None) -> None:
        self.service = service or MMMToolService()

    def dispatch(self, request: MCPRequestEnvelope) -> MCPResponseEnvelope:
        if request.tool_name not in self._ALLOWED:
            return MCPResponseEnvelope(
                status="failed", error=f"Unknown or disallowed MCP tool: {request.tool_name}"
            )
        method = getattr(self.service, request.tool_name)
        try:
            result = method(**request.input)
        except Exception as exc:
            return MCPResponseEnvelope(
                status="failed", error=f"{type(exc).__name__}: {exc}"
            )
        if not isinstance(result, dict):
            result = {"result": result}
        return MCPResponseEnvelope(status="succeeded", result=result)


__all__ = [
    "AuthContext",
    "DomainMCPServerRegistry",
    "ExecutionLimits",
    "MCPRequestEnvelope",
    "MCPResponseEnvelope",
]
