from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any

import anyio

from .external_mcp import ExternalMCPRegistry
from .mcp_stdio_support import open_mcp_stdio_errlog


class ExternalMCPError(RuntimeError):
    pass


def _server_scope(values: Collection[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(
        value
        for raw in values
        if (value := str(raw).strip())
    )


@dataclass(frozen=True)
class MCPRouteTarget:
    minecraft_version: str
    loader: str
    mappings: str
    mapping: str

    @classmethod
    def from_value(cls, value: Any) -> MCPRouteTarget:
        if value is None:
            return cls("", "fabric", "", "")
        if isinstance(value, Mapping):
            minecraft_version = str(value.get("minecraft_version", "")).strip()
            loader = str(value.get("loader", "fabric")).strip().lower() or "fabric"
            mappings = str(
                value.get("mappings", value.get("yarn_mappings", ""))
            ).strip()
        else:
            minecraft_version = str(getattr(value, "minecraft_version", "")).strip()
            loader = str(getattr(value, "loader", "fabric")).strip().lower() or "fabric"
            mappings = str(
                getattr(value, "yarn_mappings", getattr(value, "mappings", ""))
            ).strip()
        low = mappings.casefold()
        if "yarn" in low:
            mapping = "yarn"
        elif "moj" in low or "official" in low:
            mapping = "mojmap"
        elif "intermediary" in low:
            mapping = "intermediary"
        else:
            mapping = "mojmap" if _is_new_mojang_scheme(minecraft_version) else "yarn"
        return cls(minecraft_version, loader, mappings, mapping)

    def to_dict(self) -> dict[str, str]:
        return {
            "minecraft_version": self.minecraft_version,
            "loader": self.loader,
            "mappings": self.mappings,
            "mapping": self.mapping,
        }


class ExternalMCPRouter:
    """Capability router for reviewed Minecraft MCP servers.

    The router never treats an MCP provider as authority over the approved
    PlatformLock.  It supplies that target to provider tools where supported,
    verifies the tool actually exists after MCP initialization, and rejects any
    response that explicitly reports a conflicting Minecraft target.
    """

    def __init__(
        self,
        registry: ExternalMCPRegistry | None = None,
        *,
        timeout_seconds: float = 45.0,
    ) -> None:
        self.registry = registry or ExternalMCPRegistry()
        if not 1.0 <= float(timeout_seconds) <= 600.0:
            raise ValueError("External MCP timeout must be between 1 and 600 seconds.")
        self.timeout_seconds = float(timeout_seconds)

    def capability_manifest(
        self,
        *,
        stage: str,
        target: Any = None,
        max_access: str = "read",
        allowed_server_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        resolved = MCPRouteTarget.from_value(target)
        allowed_servers = _server_scope(allowed_server_ids)
        capabilities: dict[str, list[dict[str, Any]]] = {}
        all_capabilities = sorted(
            {
                capability
                for entry in self.registry.servers.values()
                for capability in entry.get("capabilities", {})
            }
        )
        for capability in all_capabilities:
            routes = self.registry.routes(
                capability,
                stage=stage,
                minecraft_version=resolved.minecraft_version,
                loader=resolved.loader,
                max_access=max_access,
            )
            if allowed_servers is not None:
                routes = [
                    route
                    for route in routes
                    if str(route["server"]) in allowed_servers
                ]
            if routes:
                capabilities[capability] = [
                    {
                        "server": item["server"],
                        "tool": item["route"]["tool"],
                        "priority": item["priority"],
                        "access": item["route"].get("access", "read"),
                        "trust": item["entry"].get("trust", "unknown"),
                        "version_policy": item["entry"].get("version_policy", "agnostic"),
                    }
                    for item in routes
                ]
        return {
            "schema_version": "mmm/external-mcp-capability-manifest-v1",
            "stage": stage,
            "target": resolved.to_dict(),
            "capabilities": capabilities,
        }

    def invoke(
        self,
        capability: str,
        *,
        stage: str,
        arguments: Mapping[str, Any] | None = None,
        target: Any = None,
        corroborate: int = 1,
        required: bool = False,
        max_access: str = "read",
        disposable_runtime: bool = False,
        allowed_server_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        if type(corroborate) is not int or corroborate < 1 or corroborate > 4:
            raise ValueError("corroborate must be between 1 and 4.")
        resolved = MCPRouteTarget.from_value(target)
        allowed_servers = _server_scope(allowed_server_ids)
        if stage != "runtime" and max_access != "read":
            raise ExternalMCPError("Non-runtime MCP federation is read-only.")
        if stage == "runtime" and max_access in {"write", "admin"} and not disposable_runtime:
            raise ExternalMCPError(
                "Write/admin Minecraft MCP tools require a disposable runtime instance."
            )

        routes = self.registry.routes(
            capability,
            stage=stage,
            minecraft_version=resolved.minecraft_version,
            loader=resolved.loader,
            max_access=max_access,
        )
        if allowed_servers is not None:
            routes = [
                route
                for route in routes
                if str(route["server"]) in allowed_servers
            ]
        attempts: list[dict[str, Any]] = []
        successes: list[dict[str, Any]] = []
        for route in routes:
            server_name = route["server"]
            entry = route["entry"]
            route_spec = route["route"]
            if not self._configured(entry):
                attempts.append(
                    {
                        "server": server_name,
                        "tool": route_spec["tool"],
                        "status": "SKIPPED_NOT_CONFIGURED",
                    }
                )
                continue
            call_args = self._arguments_for_route(
                dict(arguments or {}), route_spec, resolved
            )
            try:
                called = self._call_provider(
                    server_name,
                    entry,
                    tool=str(route_spec["tool"]),
                    arguments=call_args,
                )
                self._validate_reported_target(called["result"], route_spec, resolved)
                receipt = {
                    "schema_version": "mmm/external-mcp-call-receipt-v1",
                    "server": server_name,
                    "tool": route_spec["tool"],
                    "capability": capability,
                    "stage": stage,
                    "access": route_spec.get("access", "read"),
                    "trust": entry.get("trust", "unknown"),
                    "requested_target": resolved.to_dict(),
                    "server_info": called.get("server_info", {}),
                    "arguments_sha256": _sha256(call_args),
                    "result_sha256": _sha256(called["result"]),
                    "result": called["result"],
                    "status": "PASS",
                }
                attempts.append(
                    {
                        "server": server_name,
                        "tool": route_spec["tool"],
                        "status": "PASS",
                    }
                )
                successes.append(receipt)
                if len(successes) >= corroborate:
                    break
            except Exception as exc:  # noqa: BLE001 - route failure must fall back
                attempts.append(
                    {
                        "server": server_name,
                        "tool": route_spec["tool"],
                        "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        status = "PASS" if len(successes) >= corroborate else (
            "PARTIAL" if successes else "UNAVAILABLE"
        )
        bundle = {
            "schema_version": "mmm/external-mcp-evidence-bundle-v1",
            "capability": capability,
            "stage": stage,
            "target": resolved.to_dict(),
            "required_corroboration": corroborate,
            "status": status,
            "evidence": successes,
            "attempts": attempts,
        }
        bundle["bundle_sha256"] = _sha256(bundle)
        if required and status != "PASS":
            raise ExternalMCPError(
                f"Required MCP capability {capability!r} was not satisfied: "
                + json.dumps(attempts, ensure_ascii=False)
            )
        return bundle

    def _call_provider(
        self,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        async def run() -> dict[str, Any]:
            return await self._call_provider_async(
                server_name, entry, tool=tool, arguments=arguments
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return anyio.run(run)

        # Each call owns an independent MCP transport/session, so serializing the
        # complete provider I/O behind one router lock only adds head-of-line latency.
        value: dict[str, Any] = {}
        error: list[BaseException] = []

        def worker() -> None:
            try:
                value["result"] = anyio.run(run)
            except BaseException as exc:  # noqa: BLE001  # pragma: no cover - bridge
                error.append(exc)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout_seconds + 5.0)
        if thread.is_alive():
            raise ExternalMCPError(
                f"External MCP {server_name} exceeded the synchronous bridge timeout."
            )
        if error:
            raise ExternalMCPError(str(error[0])) from error[0]
        return value["result"]

    async def _call_provider_async(
        self,
        server_name: str,
        entry: Mapping[str, Any],
        *,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except Exception as exc:  # pragma: no cover - dependency failure
            raise ExternalMCPError("The pinned MCP Python client is unavailable.") from exc

        transport = entry.get("transport")
        with anyio.fail_after(self.timeout_seconds):
            if transport == "stdio":
                command = entry.get("command")
                if not isinstance(command, list) or not command:
                    raise ExternalMCPError(f"{server_name} has no stdio command.")
                params = StdioServerParameters(
                    command=str(command[0]),
                    args=[str(value) for value in command[1:]],
                    env=self._child_env(entry),
                )
                with open_mcp_stdio_errlog() as errlog:
                    async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            return await self._initialized_call(session, tool, arguments)
            if transport == "streamable_http":
                url = self._server_url(entry)
                if not url:
                    raise ExternalMCPError(f"{server_name} has no configured HTTP MCP URL.")
                async with streamable_http_client(url) as (
                    read_stream,
                    write_stream,
                    _,
                ), ClientSession(read_stream, write_stream) as session:
                    return await self._initialized_call(session, tool, arguments)
        raise ExternalMCPError(
            f"Federation does not invoke transport {transport!r} for {server_name}."
        )

    @staticmethod
    async def _initialized_call(
        session: Any,
        tool: str,
        arguments: Mapping[str, Any],
    ) -> dict[str, Any]:
        initialized = await session.initialize()
        listed = await session.list_tools()
        available = {str(item.name) for item in getattr(listed, "tools", ())}
        if tool not in available:
            raise ExternalMCPError(
                f"Provider does not expose reviewed tool {tool!r}; available={sorted(available)}"
            )
        raw = await session.call_tool(tool, arguments=dict(arguments))
        if bool(getattr(raw, "isError", getattr(raw, "is_error", False))):
            raise ExternalMCPError("External MCP tool returned an MCP error result.")
        return {
            "server_info": _jsonable(getattr(initialized, "serverInfo", None)),
            "result": _normalize_tool_result(raw),
        }

    @staticmethod
    def _arguments_for_route(
        arguments: dict[str, Any],
        route: Mapping[str, Any],
        target: MCPRouteTarget,
    ) -> dict[str, Any]:
        values = target.to_dict()
        for canonical, provider_name in route.get("target_args", {}).items():
            value = values.get(canonical, "")
            if value and provider_name not in arguments:
                arguments[str(provider_name)] = value
        return arguments

    @staticmethod
    def _validate_reported_target(
        result: Mapping[str, Any],
        route: Mapping[str, Any],
        target: MCPRouteTarget,
    ) -> None:
        if not target.minecraft_version:
            return
        explicit_fields = route.get("response_target_fields", [])
        tool = str(route.get("tool", "")).strip()
        authoritative = bool(explicit_fields) or bool(route.get("require_reported_target")) or tool in {
            "server_get_status",
            "client_get_status",
        }
        if not authoritative:
            # Search/docs results routinely contain historical Minecraft versions.
            # Those references are evidence content, not provider runtime authority.
            return
        reported = _collect_target_values(result, explicit_fields)
        if not reported:
            raise ExternalMCPError(
                f"External MCP authoritative tool {tool!r} did not report a Minecraft target."
            )
        conflicts = sorted(
            value for value in reported
            if value and value != target.minecraft_version
        )
        if conflicts:
            raise ExternalMCPError(
                "External MCP reported a Minecraft target that conflicts with the approved "
                f"PlatformLock: expected {target.minecraft_version!r}, got {conflicts!r}."
            )

    @staticmethod
    def _configured(entry: Mapping[str, Any]) -> bool:
        for name in entry.get("required_env", []):
            if not os.environ.get(str(name), "").strip():
                return False
        status = entry.get("status")
        if status == "configuration_required":
            url_env = str(entry.get("url_env", "")).strip()
            # A localhost default is intentionally probe-able in runtime, while a
            # provider with no URL/command must be explicitly configured.
            return bool(
                (url_env and os.environ.get(url_env, "").strip())
                or entry.get("default_url")
                or entry.get("command")
            )
        return status in {"enabled", "optional"}

    @staticmethod
    def _server_url(entry: Mapping[str, Any]) -> str:
        env_name = str(entry.get("url_env", "")).strip()
        if env_name:
            value = os.environ.get(env_name, "").strip()
            if value:
                return value
        return str(entry.get("default_url", "")).strip()

    @staticmethod
    def _child_env(entry: Mapping[str, Any]) -> dict[str, str]:
        safe_names = {
            "PATH",
            "HOME",
            "USER",
            "TMPDIR",
            "TMP",
            "TEMP",
            "JAVA_HOME",
            "XDG_CACHE_HOME",
            "NPM_CONFIG_CACHE",
            "COLAB_RELEASE_TAG",
        }
        safe_names.update(str(value) for value in entry.get("required_env", []))
        env = {
            name: value
            for name in safe_names
            if (value := os.environ.get(name)) is not None
        }
        configured = entry.get("env", {})
        if isinstance(configured, Mapping):
            env.update({str(key): str(value) for key, value in configured.items()})
        return env


def _normalize_tool_result(raw: Any) -> dict[str, Any]:
    structured = getattr(raw, "structuredContent", None)
    if structured is None:
        structured = getattr(raw, "structured_content", None)
    structured_value = _jsonable(structured)
    texts: list[str] = []
    resources: list[Any] = []
    for item in getattr(raw, "content", ()) or ():
        text = getattr(item, "text", None)
        if isinstance(text, str):
            texts.append(text)
        else:
            resources.append(_jsonable(item))
    parsed_text: Any = None
    if len(texts) == 1:
        try:
            parsed_text = json.loads(texts[0])
        except json.JSONDecodeError:
            parsed_text = None
    return {
        "structured": structured_value,
        "parsed_text": _jsonable(parsed_text),
        "text": texts,
        "other_content": resources,
    }


def _collect_target_values(
    value: Any,
    extra_fields: Any = (),
) -> set[str]:
    field_names = {
        "minecraft_version",
        "minecraftVersion",
        "mcVersion",
        "mc_version",
    }
    if isinstance(extra_fields, list):
        field_names.update(str(field) for field in extra_fields)
    found: set[str] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, Mapping):
            for key, child in node.items():
                if str(key) in field_names and isinstance(child, (str, int, float)):
                    found.add(str(child).strip())
                elif str(key) in {"target", "platform", "metadata", "result", "structured", "parsed_text"} or isinstance(child, (Mapping, list, tuple)):
                    walk(child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for child in node[:100]:
                walk(child, depth + 1)

    walk(value)
    return found


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    dumper = getattr(value, "model_dump", None)
    if callable(dumper):
        try:
            return _jsonable(dumper(mode="json"))
        except TypeError:
            return _jsonable(dumper())
    return str(value)


def _sha256(value: Any) -> str:
    raw = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _is_new_mojang_scheme(version: str) -> bool:
    try:
        major = int(version.split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return major >= 26
