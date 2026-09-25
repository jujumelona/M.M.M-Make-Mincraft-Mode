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


def _with_capability(schema: Mapping[str, Any], capability: str, state: Any) -> Mapping[str, Any]:
    projected = deepcopy(schema)
    function = projected.get("function")
    parameters = function.get("parameters") if isinstance(function, Mapping) else None
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    field = properties.get("capability") if isinstance(properties, Mapping) else None
    if isinstance(field, dict):
        field["enum"] = [capability]
    provider_schema = getattr(state, "_external_mcp_provider_schema", None)
    if (
        _tool_name(schema) == "external_mcp_call"
        and getattr(state, "_external_mcp_schema_capability", "") == capability
        and isinstance(properties, dict) and isinstance(provider_schema, Mapping)
    ):
        nested = deepcopy(dict(provider_schema))

        def relocate_refs(value: Any) -> None:
            if isinstance(value, dict):
                if "$id" in value:
                    return  # References below this resource retain their own base URI.
                ref = value.get("$ref")
                if ref == "#":
                    value["$ref"] = "#/properties/arguments"
                elif isinstance(ref, str) and ref.startswith("#/"):
                    value["$ref"] = "#/properties/arguments/" + ref[2:]
                for child in value.values():
                    relocate_refs(child)
            elif isinstance(value, list):
                for child in value:
                    relocate_refs(child)

        relocate_refs(nested)
        properties["arguments"] = nested
    return projected


def _preferred_capability(state: Any, repair_route: str | None) -> str:
    available = set(getattr(state, "_external_mcp_recovery_capabilities", ()) or ())
    available.difference_update(getattr(state, "_external_mcp_completed_capabilities", ()) or ())
    order = _RECOVERY_ORDER.get(str(repair_route or "").strip(), _DEFAULT_ORDER)
    diagnostics = getattr(state, "latest_verifier_errors", ()) or ()
    if any(isinstance(error, Mapping) and "net.fabricmc." in str(error.get("message", ""))
           for error in diagnostics):
        # Fabric API classes are not in the decompiled vanilla Minecraft tree.
        order = ("official_mod_docs",) + tuple(name for name in order if name != "official_mod_docs")
    return next((name for name in order if name in available), "")


def recovery_state_snapshot(
    state: Any,
    repair_route: str | None,
) -> dict[str, Any]:
    """Return a small, log-safe view of the external MCP recovery frontier."""

    available = tuple(
        str(item)
        for item in (getattr(state, "_external_mcp_recovery_capabilities", ()) or ())
        if str(item)
    )
    completed = tuple(
        sorted(
            str(item)
            for item in (getattr(state, "_external_mcp_completed_capabilities", ()) or ())
            if str(item)
        )
    )
    return {
        "capabilities_seen": bool(
            getattr(state, "_external_mcp_capabilities_seen", False)
        ),
        "available_capabilities": list(available),
        "completed_capabilities": list(completed),
        "bound_schema_capability": str(
            getattr(state, "_external_mcp_schema_capability", "") or ""
        )
        or None,
        "next_capability": _preferred_capability(state, repair_route) or None,
        "repair_route": str(repair_route or "") or None,
    }


def constrain_recovery_tools(
    phase_tools: Sequence[Mapping[str, Any]],
    *,
    state: Any,
    repair_route: str | None,
    available_tools: Sequence[Mapping[str, Any]] = (),
) -> tuple[Mapping[str, Any], ...]:
    """Project schema/call tools to one host-selected discovered capability."""

    tools = tuple(phase_tools)
    phase = getattr(state, "phase", None)
    phase_name = getattr(phase, "value", phase)
    external_frontier = not tools or (
        len(tools) == 1
        and _tool_name(tools[0]) in {"external_mcp_schema", "external_mcp_call"}
    )
    if (
        available_tools
        and phase_name in {"OBSERVE", "RECOVER"}
        and bool(getattr(state, "_external_mcp_capabilities_seen", False))
        and external_frontier
    ):
        # A generic tool attempt is not a capability attempt. Each discovered
        # provider route gets one schema/call pair; exhaustion remains finite.
        capability = _preferred_capability(state, repair_route)
        if not capability:
            return ()
        bound = getattr(state, "_external_mcp_schema_capability", "")
        name = "external_mcp_call" if bound == capability else "external_mcp_schema"
        schema = next((item for item in available_tools if _tool_name(item) == name), None)
        return (_with_capability(schema, capability, state),) if schema is not None else ()
    if len(tools) != 1:
        return tools
    name = _tool_name(tools[0])
    if name == "external_mcp_schema":
        if not bool(getattr(state, "_external_mcp_capabilities_seen", False)):
            return tools
        capability = _preferred_capability(state, repair_route)
        return (_with_capability(tools[0], capability, state),) if capability else ()
    if name == "external_mcp_call":
        capability = str(
            getattr(state, "_external_mcp_schema_capability", "") or ""
        ).strip()
        return (_with_capability(tools[0], capability, state),) if capability else ()
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
        state._external_mcp_capabilities_seen = True
        state._external_mcp_recovery_capabilities = reviewed
        state._external_mcp_schema_capability = ""
        state._external_mcp_provider_schema = None
        state._external_mcp_completed_capabilities = set()
        return
    if name not in {"external_mcp_schema", "external_mcp_call"}:
        return
    result = payload.get("result")
    status = str(result.get("status") or "").strip() if isinstance(result, Mapping) else ""
    capability = str(getattr(call, "arguments", {}).get("capability") or "").strip()
    available = set(getattr(state, "_external_mcp_recovery_capabilities", ()) or ())
    if capability not in available:
        return
    if payload.get("failure_code") == "EXTERNAL_MCP_ARGUMENTS_INVALID":
        return
    if name == "external_mcp_call" or status != "PASS":
        completed = set(getattr(state, "_external_mcp_completed_capabilities", ()) or ())
        completed.add(capability)
        state._external_mcp_completed_capabilities = completed
        state._external_mcp_schema_capability = ""
        state._external_mcp_provider_schema = None
        return
    state._external_mcp_schema_capability = capability
    schema = result.get("input_schema") if isinstance(result, Mapping) else None
    state._external_mcp_provider_schema = deepcopy(dict(schema)) if isinstance(schema, Mapping) else None
    if isinstance(state._external_mcp_provider_schema, dict):
        # The router supplies these coordinates after model validation. They must
        # not become required model arguments merely because the provider needs them.
        injected = set((result.get("target_args_injected_by_router") or {}).values())
        properties = state._external_mcp_provider_schema.get("properties", {})
        for field in injected:
            properties.pop(field, None)
        required = state._external_mcp_provider_schema.get("required")
        if isinstance(required, list):
            state._external_mcp_provider_schema["required"] = [name for name in required if name not in injected]


__all__ = [
    "constrain_recovery_tools",
    "record_discovery",
    "recovery_state_snapshot",
]
