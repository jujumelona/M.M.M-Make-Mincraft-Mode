from __future__ import annotations

"""Bind recovery-time external MCP selection to discovered reviewed capabilities."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

_RECOVERY_ORDER = {
    "official_api": (
        "source_search",
        "mapping_resolution",
        "registry_lookup",
        "official_mod_docs",
        "vanilla_knowledge",
        "version_diff",
        "mod_examples",
        "mod_jar_analysis",
    ),
    "compatibility": (
        "official_mod_docs",
        "mod_examples",
        "source_search",
        "version_diff",
        "mapping_resolution",
        "mod_jar_analysis",
    ),
}
_DEFAULT_ORDER = (
    "source_search",
    "official_mod_docs",
    "mapping_resolution",
    "registry_lookup",
    "vanilla_knowledge",
    "version_diff",
    "mod_examples",
    "mod_jar_analysis",
)


def _tool_name(schema: Mapping[str, Any]) -> str:
    function = schema.get("function")
    return str(function.get("name") or "").strip() if isinstance(function, Mapping) else ""


def _with_capability(schema: Mapping[str, Any], capability: str) -> Mapping[str, Any]:
    projected = deepcopy(schema)
    function = projected.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    field = properties.get("capability") if isinstance(properties, Mapping) else None
    if isinstance(field, dict):
        field["enum"] = [capability]
    return projected


def _preferred_capability(state: Any, repair_route: str | None) -> str:
    available = set(getattr(state, "_external_mcp_recovery_capabilities", ()) or ())
    order = _RECOVERY_ORDER.get(str(repair_route or "").strip(), _DEFAULT_ORDER)
    return next((name for name in order if name in available), "")


def constrain_recovery_tools(
    phase_tools: Sequence[Mapping[str, Any]],
    *,
    state: Any,
    repair_route: str | None,
) -> tuple[Mapping[str, Any], ...]:
    """Project schema/call tools to one host-selected discovered capability."""

    tools = tuple(phase_tools)
    if len(tools) != 1:
        return tools
    name = _tool_name(tools[0])
    if name == "external_mcp_schema":
        if not bool(getattr(state, "_external_mcp_capabilities_seen", False)):
            return tools
        capability = _preferred_capability(state, repair_route)
        return (_with_capability(tools[0], capability),) if capability else ()
    if name == "external_mcp_call":
        capability = str(
            getattr(state, "_external_mcp_schema_capability", "") or ""
        ).strip()
        return (_with_capability(tools[0], capability),) if capability else ()
    return tools


def record_discovery(
    state: Any,
    call: Any,
    payload: Mapping[str, Any],
    *,
    external_rag_capability: Callable[[Mapping[str, Any]], str],
) -> None:
    """Remember only reviewed retrieval capabilities and one successful schema owner."""

    name = str(getattr(call, "name", "") or "").strip()
    if name == "external_mcp_capabilities":
        result = payload.get("result")
        capabilities = result.get("capabilities") if isinstance(result, Mapping) else None
        reviewed = ()
        if isinstance(capabilities, Mapping):
            reviewed = tuple(
                sorted(
                    capability
                    for raw in capabilities
                    if (
                        capability := external_rag_capability(
                            {"capability": str(raw)}
                        )
                    )
                )
            )
        setattr(state, "_external_mcp_capabilities_seen", True)
        setattr(state, "_external_mcp_recovery_capabilities", reviewed)
        setattr(state, "_external_mcp_schema_capability", "")
        return
    if name != "external_mcp_schema":
        return
    result = payload.get("result")
    status = str(result.get("status") or "").strip() if isinstance(result, Mapping) else ""
    capability = str(getattr(call, "arguments", {}).get("capability") or "").strip()
    available = set(getattr(state, "_external_mcp_recovery_capabilities", ()) or ())
    setattr(
        state,
        "_external_mcp_schema_capability",
        capability if status == "PASS" and capability in available else "",
    )


__all__ = ["constrain_recovery_tools", "record_discovery"]
