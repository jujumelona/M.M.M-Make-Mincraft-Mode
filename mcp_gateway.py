"""Compatibility facade backed by the real core and production MCP tool services."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from minecraft_mod_ai.mcp_tools import MMMToolService
from minecraft_mod_ai.production_tools import ProductionToolService


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


_CORE_TOOLS = frozenset(
    {
        "plan_game",
        "plan_complete_game",
        "revise_plan",
        "approve_plan",
        "approve_complete_plan",
        "read_quality_contract",
        "quality_status",
        "execute_complete_project",
        "apply_source_patch",
        "repair_project",
        "search_project_rag",
        "inspect_existing_mod",
        "generate_fabric_project",
        "generate_assets",
        "run_static_validation",
        "run_gradle_build",
        "run_gametest",
        "inspect_jar",
        "package_release",
    }
)
_PRODUCTION_TOOLS = frozenset(
    {
        "index_project_rag",
        "search_code_rag",
        "java_diagnostics",
        "java_workspace_symbols",
        "blockbench_list_tools",
        "blockbench_execute",
        "generate_geckolib_entity",
        "generate_system_plugin",
        "runtime_prepare_instance",
        "runtime_start_server",
        "runtime_start_client",
        "runtime_send_command",
        "runtime_logs",
        "runtime_register_screenshot",
        "runtime_status",
        "runtime_stop",
        "mineflayer_connect",
        "mineflayer_status",
        "mineflayer_walk_to",
        "mineflayer_interact_block",
        "mineflayer_inventory",
        "mineflayer_disconnect",
        "run_model_smoke",
        "record_training_trace",
        "export_training_dataset",
        "system_plugin_ids",
    }
)


class DomainMCPServerRegistry:
    _ALLOWED = _CORE_TOOLS | _PRODUCTION_TOOLS

    def __init__(
        self,
        service: MMMToolService | None = None,
        production_service: ProductionToolService | None = None,
    ) -> None:
        self.service = service or MMMToolService()
        self.production_service = production_service or ProductionToolService()

    def dispatch(self, request: MCPRequestEnvelope) -> MCPResponseEnvelope:
        if request.tool_name not in self._ALLOWED:
            return MCPResponseEnvelope(
                status="failed",
                error=f"Unknown or disallowed MCP tool: {request.tool_name}",
            )
        target = self.service if request.tool_name in _CORE_TOOLS else self.production_service
        method = getattr(target, request.tool_name)
        try:
            result = method(**request.input)
        except Exception as exc:
            return MCPResponseEnvelope(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
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
