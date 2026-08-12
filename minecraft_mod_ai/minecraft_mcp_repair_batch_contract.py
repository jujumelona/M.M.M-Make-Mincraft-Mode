from __future__ import annotations

import asyncio
import json
import re
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

import anyio

from . import external_mcp_router as router_api
from .external_mcp_router import ExternalMCPRouter, MCPRouteTarget
from .platform_catalog import adapter_from_project


_TECHNICAL_SYMBOL = re.compile(
    r"\b(?:[a-zA-Z_$][\w$]*\.){1,8}[A-Z_$][\w$]*\b|\b[A-Z][A-Za-z0-9_$]{3,}\b"
)


def install(repair_engine_module: Any) -> None:
    """Replace per-call repair federation with provider-batched validation."""

    cls = repair_engine_module.RepairEngine
    current = cls._evidence
    if getattr(current, "_mmm_external_mcp_batched", False):
        return

    # The federation wrapper was installed immediately before this contract and used
    # functools.wraps, so __wrapped__ is the deterministic JDT/Gradle evidence path.
    base_evidence = getattr(current, "__wrapped__", current)

    @wraps(base_evidence)
    def evidence(self: Any, root: Path, *, run_gametest: bool) -> dict[str, Any]:
        base = base_evidence(self, root, run_gametest=run_gametest)
        try:
            adapter = adapter_from_project(root)
        except Exception as exc:
            enriched = dict(base)
            enriched["external_mcp"] = {
                "status": "SKIPPED_NO_PLATFORM_LOCK",
                "error": f"{type(exc).__name__}: {exc}",
            }
            return enriched

        router = ExternalMCPRouter(timeout_seconds=180.0)
        workspace_bundle = _invoke_safe(
            router,
            "workspace_validation",
            {
                "task": "project-summary",
                "subject": {
                    "kind": "workspace",
                    "projectPath": str(root),
                    "discover": ["mixins", "access-wideners", "access-transformers"],
                },
                "preferProjectVersion": True,
                "preferProjectMapping": True,
            },
            adapter,
        )

        minecraft_dev_requests: list[tuple[str, dict[str, Any]]] = []
        symbols = _technical_symbols(_diagnostic_text(base))
        if symbols:
            minecraft_dev_requests.append(
                (
                    "source_search",
                    {
                        "query": "|".join(re.escape(value) for value in symbols),
                        "searchType": "all",
                        "limit": 50,
                    },
                )
            )

        if _contains_mixins(root):
            # analyze_mixin explicitly accepts a directory. One directory-level call
            # avoids N process launches and lets the provider inspect related files.
            minecraft_dev_requests.append(
                ("mixin_validation", {"source": str(root)})
            )

        access_wideners = _access_wideners(root)
        for path in access_wideners:
            minecraft_dev_requests.append(
                ("access_widener_validation", {"content": str(path)})
            )

        access_transformers = _access_transformers(root)
        for path in access_transformers:
            others = [str(value) for value in access_transformers if value != path]
            arguments: dict[str, Any] = {"content": str(path)}
            if others:
                arguments["extraFiles"] = others
            minecraft_dev_requests.append(
                ("access_transformer_validation", arguments)
            )

        dev_bundle = _invoke_minecraft_dev_batch(
            router,
            minecraft_dev_requests,
            target=adapter,
            stage="quality",
        )
        routes = [
            {
                "capability": "workspace_validation",
                "bundle": _compact(workspace_bundle),
            }
        ]
        for result in dev_bundle["results"]:
            routes.append(
                {
                    "capability": result["capability"],
                    "bundle": _compact(result["bundle"]),
                }
            )
        enriched = dict(base)
        enriched["external_mcp"] = {
            "schema_version": "mmm/repair-external-mcp-evidence-v2",
            "target": _target_dict(adapter),
            "session_reuse": {
                "minecraft_dev": True,
                "initialize_count": dev_bundle["initialize_count"],
                "tool_call_count": len(minecraft_dev_requests),
            },
            "routes": routes,
            "authoritative_gates": ["JDT", "Gradle", "GameTest"],
        }
        return enriched

    evidence._mmm_external_mcp_batched = True
    cls._evidence = evidence


def _invoke_minecraft_dev_batch(
    router: ExternalMCPRouter,
    requests: list[tuple[str, dict[str, Any]]],
    *,
    target: Any,
    stage: str,
) -> dict[str, Any]:
    if not requests:
        return {"initialize_count": 0, "results": []}
    resolved = MCPRouteTarget.from_value(target)
    entry = router.registry.server("minecraft-dev")
    planned: list[dict[str, Any]] = []
    for capability, arguments in requests:
        matches = [
            row
            for row in router.registry.routes(
                capability,
                stage=stage,
                minecraft_version=resolved.minecraft_version,
                loader=resolved.loader,
                max_access="read",
            )
            if row["server"] == "minecraft-dev"
        ]
        if not matches:
            planned.append(
                {
                    "capability": capability,
                    "route": None,
                    "arguments": arguments,
                    "error": "minecraft-dev capability is unavailable for the approved target",
                }
            )
            continue
        route = matches[0]["route"]
        planned.append(
            {
                "capability": capability,
                "route": route,
                "arguments": router._arguments_for_route(
                    dict(arguments), route, resolved
                ),
            }
        )

    valid = [item for item in planned if item.get("route") is not None]
    if not valid:
        return {
            "initialize_count": 0,
            "results": [
                {
                    "capability": item["capability"],
                    "bundle": _unavailable(item["capability"], stage, item.get("error", "unavailable")),
                }
                for item in planned
            ],
        }

    try:
        called = _run_sync(
            _batch_session_call,
            router,
            entry,
            valid,
            timeout_seconds=router.timeout_seconds,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return {
            "initialize_count": 0,
            "results": [
                {
                    "capability": item["capability"],
                    "bundle": _unavailable(item["capability"], stage, error),
                }
                for item in planned
            ],
        }

    called_iter = iter(called)
    results: list[dict[str, Any]] = []
    for item in planned:
        capability = item["capability"]
        route = item.get("route")
        if route is None:
            results.append(
                {
                    "capability": capability,
                    "bundle": _unavailable(capability, stage, item.get("error", "unavailable")),
                }
            )
            continue
        response = next(called_iter)
        if response.get("error"):
            results.append(
                {
                    "capability": capability,
                    "bundle": _unavailable(capability, stage, str(response["error"])),
                }
            )
            continue
        result = response["result"]
        try:
            router._validate_reported_target(result, route, resolved)
        except Exception as exc:
            results.append(
                {
                    "capability": capability,
                    "bundle": _unavailable(
                        capability,
                        stage,
                        f"{type(exc).__name__}: {exc}",
                    ),
                }
            )
            continue
        receipt = {
            "schema_version": "mmm/external-mcp-call-receipt-v1",
            "server": "minecraft-dev",
            "tool": route["tool"],
            "capability": capability,
            "stage": stage,
            "access": route.get("access", "read"),
            "trust": entry.get("trust", "unknown"),
            "requested_target": resolved.to_dict(),
            "server_info": response.get("server_info", {}),
            "arguments_sha256": router_api._sha256(item["arguments"]),
            "result_sha256": router_api._sha256(result),
            "result": result,
            "status": "PASS",
        }
        bundle = {
            "schema_version": "mmm/external-mcp-evidence-bundle-v1",
            "capability": capability,
            "stage": stage,
            "target": resolved.to_dict(),
            "required_corroboration": 1,
            "status": "PASS",
            "evidence": [receipt],
            "attempts": [
                {
                    "server": "minecraft-dev",
                    "tool": route["tool"],
                    "status": "PASS",
                }
            ],
        }
        bundle["bundle_sha256"] = router_api._sha256(bundle)
        results.append({"capability": capability, "bundle": bundle})
    return {"initialize_count": 1, "results": results}


async def _batch_session_call(
    router: ExternalMCPRouter,
    entry: Mapping[str, Any],
    planned: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    command = entry.get("command")
    if not isinstance(command, list) or not command:
        raise RuntimeError("minecraft-dev has no stdio command")
    params = StdioServerParameters(
        command=str(command[0]),
        args=[str(value) for value in command[1:]],
        env=router._child_env(entry),
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            listed = await session.list_tools()
            available = {str(item.name) for item in getattr(listed, "tools", ())}
            server_info = router_api._jsonable(getattr(initialized, "serverInfo", None))
            results: list[dict[str, Any]] = []
            for item in planned:
                tool = str(item["route"]["tool"])
                if tool not in available:
                    results.append(
                        {
                            "error": f"Provider does not expose reviewed tool {tool!r}",
                            "server_info": server_info,
                        }
                    )
                    continue
                try:
                    raw = await session.call_tool(tool, arguments=item["arguments"])
                    if bool(getattr(raw, "isError", getattr(raw, "is_error", False))):
                        raise RuntimeError("MCP tool returned an error result")
                    results.append(
                        {
                            "server_info": server_info,
                            "result": router_api._normalize_tool_result(raw),
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "server_info": server_info,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            return results


def _run_sync(async_fn: Any, *args: Any, timeout_seconds: float) -> Any:
    async def runner():
        with anyio.fail_after(timeout_seconds):
            return await async_fn(*args)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return anyio.run(runner)

    value: dict[str, Any] = {}
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            value["result"] = anyio.run(runner)
        except BaseException as exc:  # pragma: no cover - event-loop bridge
            errors.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_seconds + 5.0)
    if thread.is_alive():
        raise RuntimeError("batched minecraft-dev MCP session exceeded timeout")
    if errors:
        raise RuntimeError(str(errors[0])) from errors[0]
    return value["result"]


def _invoke_safe(
    router: ExternalMCPRouter,
    capability: str,
    arguments: Mapping[str, Any],
    target: Any,
) -> dict[str, Any]:
    try:
        return router.invoke(
            capability,
            stage="quality",
            arguments=arguments,
            target=target,
            max_access="read",
            required=False,
        )
    except Exception as exc:
        return _unavailable(
            capability,
            "quality",
            f"{type(exc).__name__}: {exc}",
        )


def _unavailable(capability: str, stage: str, error: str) -> dict[str, Any]:
    return {
        "schema_version": "mmm/external-mcp-evidence-bundle-v1",
        "capability": capability,
        "stage": stage,
        "status": "UNAVAILABLE",
        "evidence": [],
        "attempts": [{"status": "ERROR", "error": error}],
    }


def _compact(bundle: Mapping[str, Any], limit: int = 6 * 1024) -> dict[str, Any]:
    value = {
        "schema_version": bundle.get("schema_version"),
        "capability": bundle.get("capability"),
        "stage": bundle.get("stage"),
        "target": bundle.get("target"),
        "status": bundle.get("status"),
        "attempts": bundle.get("attempts", []),
        "bundle_sha256": bundle.get("bundle_sha256", ""),
        "evidence": [],
    }
    for receipt in bundle.get("evidence", []):
        if not isinstance(receipt, Mapping):
            continue
        result = json.dumps(receipt.get("result", {}), ensure_ascii=False, sort_keys=True, default=str)
        value["evidence"].append(
            {
                key: receipt.get(key)
                for key in (
                    "server",
                    "tool",
                    "capability",
                    "trust",
                    "requested_target",
                    "server_info",
                    "arguments_sha256",
                    "result_sha256",
                    "status",
                )
            }
            | {"result_excerpt": result[:2048]}
        )
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) >= limit:
            break
    return value


def _technical_symbols(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(_TECHNICAL_SYMBOL.findall(text))))


def _diagnostic_text(evidence: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for item in evidence.get("diagnostics", {}).get("diagnostics", []):
        if isinstance(item, Mapping):
            parts.append(str(item.get("message", "")))
            parts.append(str(item.get("code", "")))
    build = evidence.get("build", {})
    if isinstance(build, Mapping):
        parts.append(str(build.get("error", "")))
    return "\n".join(parts)


def _contains_mixins(root: Path) -> bool:
    for path in root.rglob("*.java"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "@Mixin" in text or "org.spongepowered.asm.mixin" in text:
            return True
    return False


def _access_wideners(root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in root.rglob("*.accesswidener")
            if path.is_file() and not path.is_symlink()
        )
    )


def _access_transformers(root: Path) -> tuple[Path, ...]:
    values: list[Path] = []
    for path in root.rglob("*.cfg"):
        if not path.is_file() or path.is_symlink():
            continue
        normalized = path.as_posix().casefold()
        if "accesstransformer" in normalized or "/meta-inf/accesstransformer.cfg" in normalized:
            values.append(path)
    return tuple(sorted(values))


def _target_dict(adapter: Any) -> dict[str, str]:
    return {
        "minecraft_version": str(adapter.minecraft_version),
        "loader": str(adapter.loader),
        "mappings": str(getattr(adapter, "yarn_mappings", "")),
        "java_version": str(getattr(adapter, "java_version", "")),
    }
