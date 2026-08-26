from __future__ import annotations

"""Fail closed when MCP discovery exposes ambiguous or malformed tool schemas.

The model-facing tool surface is only trustworthy if the raw MCP ``tools/list``
response has one owner per name and every callable tool exposes an object input
schema. Do not repair malformed provider schemas into permissive empty objects:
that would make parser validation weaker than the tool that actually executes.

First-party schema caches are also bound to the exact child-process environment.
A stage-only cache is unsafe because the MCP server can expose a different surface
when feature/configuration environment variables change during a long-lived host.
"""

import hashlib
import json
from collections.abc import Mapping
from functools import wraps
from typing import Any

import anyio

from .mcp_stdio_support import open_mcp_stdio_errlog

_RAW_LIST_MARKER = "_mmm_raw_mcp_schema_integrity_v1"
_SCHEMA_ENV_MARKER = "_mmm_mcp_schema_environment_v1"
_EXTERNAL_SCHEMA_MARKER = "_mmm_external_provider_schema_integrity_v1"
_EXTERNAL_CALL_MARKER = "_mmm_external_provider_call_integrity_v1"
_SCHEMA_ENV_ATTR = "_mmm_schema_environment_sha256"


class MCPSchemaIntegrityError(RuntimeError):
    """Raw MCP discovery cannot establish one valid schema owner for a tool."""


def validate_input_schema(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MCPSchemaIntegrityError(f"{owner} input schema must be an object")
    schema = dict(value)
    if str(schema.get("type", "")).strip() != "object":
        raise MCPSchemaIntegrityError(f"{owner} input schema root type must be 'object'")
    properties = schema.get("properties")
    if properties is not None and not isinstance(properties, Mapping):
        raise MCPSchemaIntegrityError(f"{owner} input schema properties must be an object")
    required = schema.get("required")
    if required is not None and not isinstance(required, (list, tuple)):
        raise MCPSchemaIntegrityError(f"{owner} input schema required must be an array")
    return schema


def validate_raw_tool_rows(
    rows: Any,
    *,
    surface: str,
    blocked_names: frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(tuple(rows)):
        if not isinstance(raw, Mapping):
            raise MCPSchemaIntegrityError(
                f"{surface} tools/list row {index} is not an object"
            )
        row = dict(raw)
        name = str(row.get("name", "")).strip()
        if not name:
            raise MCPSchemaIntegrityError(
                f"{surface} tools/list row {index} has no tool name"
            )
        if name in seen:
            raise MCPSchemaIntegrityError(
                f"duplicate MCP tool name {name!r} in {surface} tools/list"
            )
        seen.add(name)
        if name not in blocked_names:
            row["input_schema"] = validate_input_schema(
                row.get("input_schema"), owner=f"{surface} tool {name!r}"
            )
        result.append(row)
    return tuple(result)


def _environment_fingerprint(env: Mapping[str, Any]) -> str:
    payload = json.dumps(
        sorted((str(key), str(value)) for key, value in env.items()),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_schema_environment(runtime: Any, stage: str) -> str:
    """Bind one runtime's stage cache to the exact MCP child environment.

    Only a SHA-256 is retained. Environment values (which may include credentials)
    are never copied into a second long-lived cache structure.
    """

    selected = runtime._stage(stage)
    fingerprint = _environment_fingerprint(runtime._child_env(selected))
    with runtime._lock:
        fingerprints = getattr(runtime, _SCHEMA_ENV_ATTR, None)
        if fingerprints is None:
            fingerprints = {}
            setattr(runtime, _SCHEMA_ENV_ATTR, fingerprints)
        previous = fingerprints.get(selected)
        cache_present = (
            selected in runtime._schema_cache
            or selected in runtime._allowed_tool_cache
        )
        if previous != fingerprint and (previous is not None or cache_present):
            runtime._schema_cache.pop(selected, None)
            runtime._allowed_tool_cache.pop(selected, None)
        fingerprints[selected] = fingerprint
    return fingerprint


def _listed_tool_rows(listed: Any, *, jsonable: Any) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for item in getattr(listed, "tools", ()) or ():
        schema = getattr(item, "inputSchema", None)
        if schema is None:
            schema = getattr(item, "input_schema", None)
        rows.append(
            {
                "name": str(getattr(item, "name", "")),
                "description": str(getattr(item, "description", "") or ""),
                "input_schema": jsonable(schema),
            }
        )
    return tuple(rows)


def _unique_provider_tool(
    listed: Any,
    *,
    tool: str,
    jsonable: Any,
    error_type: type[Exception],
) -> dict[str, Any]:
    try:
        rows = validate_raw_tool_rows(
            _listed_tool_rows(listed, jsonable=jsonable),
            surface="external-provider",
        )
    except MCPSchemaIntegrityError as exc:
        raise error_type(str(exc)) from exc
    matches = [row for row in rows if row["name"] == tool]
    if not matches:
        available = sorted(row["name"] for row in rows)
        raise error_type(
            f"External provider lacks reviewed tool {tool!r}; available={available}"
        )
    # validate_raw_tool_rows already makes duplicate names impossible. Keep this
    # explicit assertion at the selection boundary so future row filtering cannot
    # silently reintroduce first-writer/last-writer semantics.
    if len(matches) != 1:
        raise error_type(f"External provider exposes ambiguous tool {tool!r}")
    return matches[0]


def install(
    agent_tool_runtime_module: Any,
    external_agent_bridge_module: Any,
    external_mcp_router_module: Any,
) -> None:
    """Install raw first-party and external-provider schema integrity checks."""

    runtime_class = agent_tool_runtime_module.AgentToolRuntime

    current_list = runtime_class._list_tools_async
    if not bool(getattr(current_list, _RAW_LIST_MARKER, False)):

        @wraps(current_list)
        async def list_tools_async(self: Any, stage: str):
            rows = await current_list(self, stage)
            return list(
                validate_raw_tool_rows(
                    rows,
                    surface=f"first-party:{str(stage).strip().lower()}",
                    blocked_names=frozenset(agent_tool_runtime_module._BLOCKED_MODEL_TOOLS),
                )
            )

        setattr(list_tools_async, _RAW_LIST_MARKER, True)
        list_tools_async.__wrapped__ = current_list  # type: ignore[attr-defined]
        runtime_class._list_tools_async = list_tools_async

    current_tool_schemas = runtime_class.tool_schemas
    if not bool(getattr(current_tool_schemas, _SCHEMA_ENV_MARKER, False)):

        @wraps(current_tool_schemas)
        def tool_schemas(self: Any, stage: str):
            ensure_schema_environment(self, stage)
            return current_tool_schemas(self, stage)

        setattr(tool_schemas, _SCHEMA_ENV_MARKER, True)
        tool_schemas.__wrapped__ = current_tool_schemas  # type: ignore[attr-defined]
        runtime_class.tool_schemas = tool_schemas

    current_provider_schema = external_agent_bridge_module._provider_schema
    if not bool(getattr(current_provider_schema, _EXTERNAL_SCHEMA_MARKER, False)):

        async def provider_schema(
            entry: Mapping[str, Any],
            *,
            tool: str,
            env: Mapping[str, str],
            url: str,
            timeout_seconds: float,
        ) -> dict[str, Any]:
            try:
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client
                from mcp.client.streamable_http import streamable_http_client
            except Exception as exc:  # pragma: no cover - dependency failure
                raise external_agent_bridge_module.ExternalAgentBridgeError(
                    "The pinned MCP Python client is unavailable"
                ) from exc

            async def read(session: Any) -> dict[str, Any]:
                await session.initialize()
                listed = await session.list_tools()
                row = _unique_provider_tool(
                    listed,
                    tool=tool,
                    jsonable=external_agent_bridge_module._jsonable,
                    error_type=external_agent_bridge_module.ExternalAgentBridgeError,
                )
                return {
                    "description": row["description"],
                    "input_schema": row["input_schema"],
                }

            transport = entry.get("transport")
            with anyio.fail_after(timeout_seconds):
                if transport == "stdio":
                    command = entry.get("command")
                    if not isinstance(command, list) or not command:
                        raise external_agent_bridge_module.ExternalAgentBridgeError(
                            "External MCP stdio command is missing"
                        )
                    params = StdioServerParameters(
                        command=str(command[0]),
                        args=[str(value) for value in command[1:]],
                        env=dict(env),
                    )
                    with open_mcp_stdio_errlog() as errlog:
                        async with stdio_client(params, errlog=errlog) as (
                            read_stream,
                            write_stream,
                        ):
                            async with ClientSession(read_stream, write_stream) as session:
                                return await read(session)
                if transport == "streamable_http":
                    if not url:
                        raise external_agent_bridge_module.ExternalAgentBridgeError(
                            "External MCP HTTP URL is missing"
                        )
                    async with streamable_http_client(url) as (
                        read_stream,
                        write_stream,
                        _,
                    ), ClientSession(read_stream, write_stream) as session:
                        return await read(session)
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "Unsupported external MCP transport for schema discovery: "
                f"{transport!r}"
            )

        setattr(provider_schema, _EXTERNAL_SCHEMA_MARKER, True)
        provider_schema.__wrapped__ = current_provider_schema  # type: ignore[attr-defined]
        external_agent_bridge_module._provider_schema = provider_schema

    current_initialized_call = external_mcp_router_module.ExternalMCPRouter._initialized_call
    if not bool(getattr(current_initialized_call, _EXTERNAL_CALL_MARKER, False)):

        async def initialized_call(
            session: Any,
            tool: str,
            arguments: Mapping[str, Any],
        ) -> dict[str, Any]:
            initialized = await session.initialize()
            listed = await session.list_tools()
            _unique_provider_tool(
                listed,
                tool=tool,
                jsonable=external_mcp_router_module._jsonable,
                error_type=external_mcp_router_module.ExternalMCPError,
            )
            raw = await session.call_tool(tool, arguments=dict(arguments))
            if bool(getattr(raw, "isError", getattr(raw, "is_error", False))):
                raise external_mcp_router_module.ExternalMCPError(
                    "External MCP tool returned an MCP error result."
                )
            return {
                "server_info": external_mcp_router_module._jsonable(
                    getattr(initialized, "serverInfo", None)
                ),
                "result": external_mcp_router_module._normalize_tool_result(raw),
            }

        setattr(initialized_call, _EXTERNAL_CALL_MARKER, True)
        initialized_call.__wrapped__ = current_initialized_call  # type: ignore[attr-defined]
        external_mcp_router_module.ExternalMCPRouter._initialized_call = staticmethod(
            initialized_call
        )


__all__ = [
    "MCPSchemaIntegrityError",
    "ensure_schema_environment",
    "install",
    "validate_input_schema",
    "validate_raw_tool_rows",
]
