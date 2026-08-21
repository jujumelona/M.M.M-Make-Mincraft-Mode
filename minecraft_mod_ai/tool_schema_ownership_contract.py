from __future__ import annotations

"""Validate the fully composed model-tool surface before it is cached or executed.

MCP discovery, first-party projections, external MCP federation and narrow model-facing
ACIs are composed by several independent owners. The final runtime boundary must never
let wrapper order decide which same-name schema wins, nor let a first-party MCP tool
shadow a reserved external-federation dispatch name.
"""

from functools import wraps
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_tool_schema_ownership_v1"


class ToolSchemaOwnershipError(RuntimeError):
    """The composed tool surface has ambiguous or incompatible ownership."""


def _schema_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _schema_parameters(schema: Any) -> Mapping[str, Any] | None:
    if not isinstance(schema, Mapping):
        return None
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return None
    parameters = function.get("parameters", {})
    return parameters if isinstance(parameters, Mapping) else None


def validate_tool_schema_surface(
    schemas: Sequence[Any],
    *,
    surface: str,
    reserved_external_schemas: Mapping[str, Mapping[str, Any]] | None = None,
    expected_parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return a validated surface with exactly one schema owner per tool name."""

    reserved = dict(reserved_external_schemas or {})
    expected = dict(expected_parameters or {})
    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(schemas):
        if not isinstance(raw, Mapping):
            raise ToolSchemaOwnershipError(
                f"{surface} tool schema at index {index} is not an object"
            )
        if str(raw.get("type", "")).strip() != "function":
            raise ToolSchemaOwnershipError(
                f"{surface} tool schema at index {index} is not a function tool"
            )
        name = _schema_name(raw)
        if not name:
            raise ToolSchemaOwnershipError(
                f"{surface} tool schema at index {index} has no function name"
            )
        if name in seen:
            raise ToolSchemaOwnershipError(
                f"duplicate tool schema name {name!r} in {surface} surface"
            )
        seen.add(name)
        parameters = _schema_parameters(raw)
        if parameters is None:
            raise ToolSchemaOwnershipError(
                f"tool {name!r} parameters schema must be an object"
            )

        if name in reserved:
            if dict(raw) != dict(reserved[name]):
                raise ToolSchemaOwnershipError(
                    f"tool {name!r} conflicts with the reserved external MCP dispatch owner"
                )
        elif name.startswith("external_mcp_"):
            raise ToolSchemaOwnershipError(
                f"tool {name!r} uses the reserved external MCP namespace without an owner"
            )

        required_parameters = expected.get(name)
        if required_parameters is not None and dict(parameters) != dict(required_parameters):
            raise ToolSchemaOwnershipError(
                f"tool {name!r} does not match its final model-facing parameter contract"
            )
        result.append(raw)
    return tuple(result)


def install(
    runtime_module: Any,
    *,
    expected_parameters: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Install one final validator around the composed AgentToolRuntime schema owner."""

    current = runtime_module.AgentToolRuntime.tool_schemas
    if bool(getattr(current, _MARKER, False)):
        return
    expected = dict(expected_parameters or {})

    @wraps(current)
    def tool_schemas(self: Any, stage: str):
        rows = tuple(current(self, stage))
        external_rows = tuple(self._external_bridge.tool_schemas(str(stage).strip().lower()))
        reserved = {
            name: row
            for row in external_rows
            if (name := _schema_name(row))
        }
        return validate_tool_schema_surface(
            rows,
            surface=f"agent-runtime:{str(stage).strip().lower()}",
            reserved_external_schemas=reserved,
            expected_parameters=expected,
        )

    setattr(tool_schemas, _MARKER, True)
    tool_schemas.__wrapped__ = current  # type: ignore[attr-defined]
    runtime_module.AgentToolRuntime.tool_schemas = tool_schemas


__all__ = [
    "ToolSchemaOwnershipError",
    "install",
    "validate_tool_schema_surface",
]
