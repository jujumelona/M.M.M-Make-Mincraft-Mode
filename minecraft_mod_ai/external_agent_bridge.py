from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Collection, Mapping

import anyio


CAPABILITIES_TOOL = "external_mcp_capabilities"
SCHEMA_TOOL = "external_mcp_schema"
CALL_TOOL = "external_mcp_call"
TOOL_NAMES = frozenset({CAPABILITIES_TOOL, SCHEMA_TOOL, CALL_TOOL})
AGENT_STAGES = frozenset({"planning", "research", "generation", "quality", "runtime"})


class ExternalAgentBridgeError(RuntimeError):
    pass


class ExternalAgentBridge:
    """Let a model discover, inspect and invoke reviewed external MCP capabilities."""

    def __init__(self, *, timeout_seconds: float = 120.0) -> None:
        if not 1.0 <= float(timeout_seconds) <= 600.0:
            raise ValueError("timeout_seconds must be between 1 and 600")
        self.timeout_seconds = float(timeout_seconds)
        self._router: Any | None = None
        self._schema_cache: dict[
            tuple[str, str, str, str, str, str, tuple[str, ...] | None], dict[str, Any]
        ] = {}
        self._lock = threading.RLock()

    @staticmethod
    def tool_schemas(stage: str) -> tuple[dict[str, Any], ...]:
        if stage not in AGENT_STAGES:
            return ()
        target_properties = {
            "minecraft_version": {
                "type": "string",
                "description": "Target Minecraft version. Defaults to the active MMM target.",
            },
            "loader": {
                "type": "string",
                "description": "Target mod loader. Defaults to fabric.",
            },
            "mappings": {
                "type": "string",
                "description": "Target mapping namespace/version. Defaults to the active MMM target.",
            },
            "max_access": {
                "type": "string",
                "enum": ["read", "write", "admin"],
                "description": (
                    "Use read outside runtime. Runtime write/admin is allowed only for "
                    "a disposable runtime instance."
                ),
            },
        }
        return (
            {
                "type": "function",
                "function": {
                    "name": CAPABILITIES_TOOL,
                    "description": (
                        "List reviewed external Minecraft MCP capabilities available in the "
                        "current stage, including source search, mappings, official mod docs, "
                        "examples, registry lookup, JAR/mixin checks, wiki knowledge and "
                        "runtime inspection when configured."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": dict(target_properties),
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": SCHEMA_TOOL,
                    "description": (
                        "Read the live input schema of the preferred external MCP tool for a "
                        "capability. Use this before calling a capability when argument names "
                        "are not already known."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "capability": {"type": "string"},
                            **target_properties,
                        },
                        "required": ["capability"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": CALL_TOOL,
                    "description": (
                        "Invoke a reviewed external Minecraft MCP capability. Platform target "
                        "arguments are injected by MMM according to the reviewed registry."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "capability": {"type": "string"},
                            "arguments": {
                                "type": "object",
                                "description": "Arguments matching external_mcp_schema.",
                                "additionalProperties": True,
                            },
                            "corroborate": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 4,
                                "default": 1,
                            },
                            "disposable_runtime": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "Set true only for the disposable test runtime when using "
                                    "external write/admin runtime capabilities."
                                ),
                            },
                            **target_properties,
                        },
                        "required": ["capability", "arguments"],
                        "additionalProperties": False,
                    },
                },
            },
        )

    def call(
        self,
        stage: str,
        name: str,
        payload: Mapping[str, Any],
        *,
        allowed_server_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        if stage not in AGENT_STAGES:
            raise ExternalAgentBridgeError(
                f"External MCP federation is unavailable in stage {stage!r}."
            )
        if name not in TOOL_NAMES:
            raise ExternalAgentBridgeError(f"Unknown external agent tool: {name!r}")
        target = _target(payload)
        max_access = str(payload.get("max_access", "read")).strip().lower() or "read"
        if max_access not in {"read", "write", "admin"}:
            raise ExternalAgentBridgeError("max_access must be read, write or admin")
        if stage != "runtime" and max_access != "read":
            raise ExternalAgentBridgeError("Non-runtime external MCP access is read-only")
        router = self._external_router()
        allowed_servers = (
            None
            if allowed_server_ids is None
            else frozenset(
                value
                for raw in allowed_server_ids
                if (value := str(raw).strip())
            )
        )

        if name == CAPABILITIES_TOOL:
            return router.capability_manifest(
                stage=stage,
                target=target,
                max_access=max_access,
                allowed_server_ids=allowed_servers,
            )

        capability = str(payload.get("capability", "")).strip()
        if not capability:
            raise ExternalAgentBridgeError("capability must not be empty")

        if name == SCHEMA_TOOL:
            key = (
                stage,
                capability,
                target["minecraft_version"],
                target["loader"],
                target["mappings"],
                max_access,
                None if allowed_servers is None else tuple(sorted(allowed_servers)),
            )
            with self._lock:
                cached = self._schema_cache.get(key)
                if cached is not None:
                    return cached
            result = self._run_async(
                self._describe_async,
                stage,
                capability,
                target,
                max_access,
                allowed_servers,
            )
            with self._lock:
                self._schema_cache[key] = result
            return result

        raw_arguments = payload.get("arguments", {})
        if raw_arguments is None:
            raw_arguments = {}
        if not isinstance(raw_arguments, Mapping):
            raise ExternalAgentBridgeError("arguments must be an object")
        corroborate = payload.get("corroborate", 1)
        if type(corroborate) is not int or not 1 <= corroborate <= 4:
            raise ExternalAgentBridgeError("corroborate must be an integer from 1 to 4")
        disposable_runtime = payload.get("disposable_runtime", False)
        if type(disposable_runtime) is not bool:
            raise ExternalAgentBridgeError("disposable_runtime must be a boolean")
        arguments = dict(raw_arguments)
        for reserved in self._reserved_target_arguments(
            router,
            capability=capability,
            stage=stage,
            target=target,
            max_access=max_access,
            allowed_server_ids=allowed_servers,
        ):
            arguments.pop(reserved, None)
        return router.invoke(
            capability,
            stage=stage,
            arguments=arguments,
            target=target,
            corroborate=corroborate,
            required=False,
            max_access=max_access,
            disposable_runtime=disposable_runtime,
            allowed_server_ids=allowed_servers,
        )

    @staticmethod
    def _reserved_target_arguments(
        router: Any,
        *,
        capability: str,
        stage: str,
        target: Mapping[str, str],
        max_access: str,
        allowed_server_ids: Collection[str] | None,
    ) -> frozenset[str]:
        routes = router.registry.routes(
            capability,
            stage=stage,
            minecraft_version=target["minecraft_version"],
            loader=target["loader"],
            max_access=max_access,
        )
        if allowed_server_ids is not None:
            allowed = frozenset(str(value) for value in allowed_server_ids)
            routes = [route for route in routes if str(route["server"]) in allowed]
        return frozenset(
            provider_name
            for route in routes
            for raw_name in route["route"].get("target_args", {}).values()
            if (provider_name := str(raw_name).strip())
        )

    def _external_router(self) -> Any:
        if self._router is None:
            from .external_mcp_router import ExternalMCPRouter

            self._router = ExternalMCPRouter(
                timeout_seconds=min(self.timeout_seconds, 120.0)
            )
        return self._router

    def _run_async(self, function: Any, *args: Any) -> Any:
        async def runner() -> Any:
            return await function(*args)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(runner)

        result: dict[str, Any] = {}
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                result["value"] = anyio.run(runner)
            except BaseException as exc:  # pragma: no cover - event-loop bridge
                errors.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout_seconds + 5.0)
        if thread.is_alive():
            raise ExternalAgentBridgeError("External MCP schema bridge timed out")
        if errors:
            raise ExternalAgentBridgeError(str(errors[0])) from errors[0]
        return result["value"]

    async def _describe_async(
        self,
        stage: str,
        capability: str,
        target: Mapping[str, str],
        max_access: str,
        allowed_server_ids: Collection[str] | None,
    ) -> dict[str, Any]:
        from .external_mcp_router import MCPRouteTarget

        router = self._external_router()
        resolved = MCPRouteTarget.from_value(target)
        routes = router.registry.routes(
            capability,
            stage=stage,
            minecraft_version=resolved.minecraft_version,
            loader=resolved.loader,
            max_access=max_access,
        )
        if allowed_server_ids is not None:
            routes = [
                route
                for route in routes
                if str(route["server"]) in allowed_server_ids
            ]
        attempts: list[dict[str, Any]] = []
        for route in routes:
            server = str(route["server"])
            entry = route["entry"]
            route_spec = route["route"]
            tool = str(route_spec["tool"])
            if not router._configured(entry):
                attempts.append(
                    {"server": server, "tool": tool, "status": "SKIPPED_NOT_CONFIGURED"}
                )
                continue
            try:
                live = await _provider_schema(
                    entry,
                    tool=tool,
                    env=router._child_env(entry),
                    url=router._server_url(entry),
                    timeout_seconds=min(self.timeout_seconds, 120.0),
                )
                return {
                    "schema_version": "mmm/external-mcp-agent-tool-schema-v1",
                    "capability": capability,
                    "stage": stage,
                    "target": resolved.to_dict(),
                    "server": server,
                    "tool": tool,
                    "access": route_spec.get("access", "read"),
                    "trust": entry.get("trust", "unknown"),
                    "target_args_injected_by_router": dict(
                        route_spec.get("target_args", {})
                    ),
                    "description": live["description"],
                    "input_schema": live["input_schema"],
                    "status": "PASS",
                }
            except Exception as exc:
                attempts.append(
                    {
                        "server": server,
                        "tool": tool,
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        return {
            "schema_version": "mmm/external-mcp-agent-tool-schema-v1",
            "capability": capability,
            "stage": stage,
            "target": resolved.to_dict(),
            "status": "UNAVAILABLE",
            "attempts": attempts,
        }


async def _provider_schema(
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
        raise ExternalAgentBridgeError("The pinned MCP Python client is unavailable") from exc

    async def read(session: Any) -> dict[str, Any]:
        await session.initialize()
        listed = await session.list_tools()
        for item in getattr(listed, "tools", ()) or ():
            if str(getattr(item, "name", "")) != tool:
                continue
            schema = getattr(item, "inputSchema", None)
            if schema is None:
                schema = getattr(item, "input_schema", None)
            return {
                "description": str(getattr(item, "description", "") or ""),
                "input_schema": _jsonable(schema),
            }
        raise ExternalAgentBridgeError(f"External provider lacks tool {tool!r}")

    transport = entry.get("transport")
    with anyio.fail_after(timeout_seconds):
        if transport == "stdio":
            command = entry.get("command")
            if not isinstance(command, list) or not command:
                raise ExternalAgentBridgeError("External MCP stdio command is missing")
            params = StdioServerParameters(
                command=str(command[0]),
                args=[str(value) for value in command[1:]],
                env=dict(env),
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    return await read(session)
        if transport == "streamable_http":
            if not url:
                raise ExternalAgentBridgeError("External MCP HTTP URL is missing")
            async with streamable_http_client(url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    return await read(session)
    raise ExternalAgentBridgeError(
        f"Unsupported external MCP transport for schema discovery: {transport!r}"
    )


def _target(payload: Mapping[str, Any]) -> dict[str, str]:
    values = {
        "minecraft_version": str(payload.get("minecraft_version", os.environ.get("MMM_MINECRAFT_VERSION", ""))).strip(),
        "loader": str(payload.get("loader", os.environ.get("MMM_LOADER", ""))).strip(),
        "mappings": str(payload.get("mappings", os.environ.get("MMM_MAPPINGS", ""))).strip(),
    }
    # Pre-target discovery is allowed, but empty coordinates stay empty. Exact
    # target-scoped federation is rebound after the host selects PlatformLock.
    return values


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _jsonable(model_dump(mode="json"))
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)
