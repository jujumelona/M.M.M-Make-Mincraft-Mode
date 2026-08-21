from __future__ import annotations

"""Bind external MCP schema discovery to the exact provider that executes it.

Historically ``external_mcp_schema`` selected the first live provider, while the
later ``external_mcp_call`` independently routed again and could fail over to a
provider with a different input schema.  That is a TOCTOU contract violation: the
model can be validated against schema A and execute tool B.  This contract keeps a
live schema binding per request scope, revalidates it immediately before execution,
and invokes exactly that reviewed server/tool pair.  Provider/schema drift requires
an explicit schema refresh instead of silent failover.
"""

import hashlib
import json
from functools import wraps
from typing import Any, Collection, Mapping

from .mcp_schema_integrity_contract import validate_input_schema

_MARKER = "_mmm_external_mcp_schema_binding_v1"
_ROUTER_MARKER = "_mmm_external_mcp_bound_invoke_v1"
_BINDINGS_ATTR = "_mmm_external_schema_bindings"


class ExternalMCPSchemaBindingError(RuntimeError):
    """The provider/tool schema observed by the model is no longer executable."""


def _server_scope(values: Collection[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(
        value
        for raw in values
        if (value := str(raw).strip())
    )


def _binding_key(
    bridge_module: Any,
    *,
    stage: str,
    payload: Mapping[str, Any],
    allowed_server_ids: Collection[str] | None,
) -> tuple[str, str, str, str, str, str, tuple[str, ...] | None]:
    target = bridge_module._target(payload)
    allowed = _server_scope(allowed_server_ids)
    return (
        stage,
        str(payload.get("capability", "")).strip(),
        target["minecraft_version"],
        target["loader"],
        target["mappings"],
        str(payload.get("max_access", "read")).strip().lower() or "read",
        None if allowed is None else tuple(sorted(allowed)),
    )


def _fingerprint(schema: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(schema),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _binding_store(bridge: Any) -> dict[Any, dict[str, Any]]:
    store = getattr(bridge, _BINDINGS_ATTR, None)
    if store is None:
        store = {}
        setattr(bridge, _BINDINGS_ATTR, store)
    return store


def _schema_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "capability": str(payload.get("capability", "")).strip(),
    }
    for name in ("minecraft_version", "loader", "mappings", "max_access"):
        if name in payload:
            result[name] = payload[name]
    return result


def install(external_agent_bridge_module: Any, external_mcp_router_module: Any) -> None:
    """Install exact provider binding and a single-route router execution primitive."""

    router_class = external_mcp_router_module.ExternalMCPRouter
    if not bool(getattr(router_class, _ROUTER_MARKER, False)):

        def invoke_bound(
            self: Any,
            capability: str,
            *,
            stage: str,
            server: str,
            tool: str,
            arguments: Mapping[str, Any] | None = None,
            target: Any = None,
            max_access: str = "read",
            disposable_runtime: bool = False,
            allowed_server_ids: Collection[str] | None = None,
        ) -> dict[str, Any]:
            resolved = external_mcp_router_module.MCPRouteTarget.from_value(target)
            allowed = _server_scope(allowed_server_ids)
            if stage != "runtime" and max_access != "read":
                raise external_mcp_router_module.ExternalMCPError(
                    "Non-runtime MCP federation is read-only."
                )
            if (
                stage == "runtime"
                and max_access in {"write", "admin"}
                and not disposable_runtime
            ):
                raise external_mcp_router_module.ExternalMCPError(
                    "Write/admin Minecraft MCP tools require a disposable runtime instance."
                )
            if allowed is not None and server not in allowed:
                raise external_mcp_router_module.ExternalMCPError(
                    f"Bound MCP server {server!r} is outside the authorized provider scope."
                )

            routes = self.registry.routes(
                capability,
                stage=stage,
                minecraft_version=resolved.minecraft_version,
                loader=resolved.loader,
                max_access=max_access,
            )
            matches = [
                route
                for route in routes
                if str(route["server"]) == server
                and str(route["route"].get("tool", "")).strip() == tool
            ]
            if len(matches) != 1:
                raise external_mcp_router_module.ExternalMCPError(
                    "Bound external MCP route is missing or ambiguous: "
                    f"server={server!r} tool={tool!r} matches={len(matches)}"
                )
            route = matches[0]
            entry = route["entry"]
            route_spec = route["route"]
            if not self._configured(entry):
                raise external_mcp_router_module.ExternalMCPError(
                    f"Bound external MCP provider {server!r} is no longer configured."
                )

            call_args = self._arguments_for_route(
                dict(arguments or {}), route_spec, resolved
            )
            called = self._call_provider(
                server,
                entry,
                tool=tool,
                arguments=call_args,
            )
            self._validate_reported_target(called["result"], route_spec, resolved)
            receipt = {
                "schema_version": "mmm/external-mcp-call-receipt-v1",
                "server": server,
                "tool": tool,
                "capability": capability,
                "stage": stage,
                "access": route_spec.get("access", "read"),
                "trust": entry.get("trust", "unknown"),
                "requested_target": resolved.to_dict(),
                "server_info": called.get("server_info", {}),
                "arguments_sha256": external_mcp_router_module._sha256(call_args),
                "result_sha256": external_mcp_router_module._sha256(called["result"]),
                "result": called["result"],
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
                    {"server": server, "tool": tool, "status": "PASS"}
                ],
            }
            bundle["bundle_sha256"] = external_mcp_router_module._sha256(bundle)
            return bundle

        setattr(router_class, "invoke_bound", invoke_bound)
        setattr(router_class, _ROUTER_MARKER, True)

    bridge_class = external_agent_bridge_module.ExternalAgentBridge
    current = bridge_class.call
    if bool(getattr(current, _MARKER, False)):
        return

    def invalidate(self: Any, key: Any) -> None:
        with self._lock:
            _binding_store(self).pop(key, None)
            self._schema_cache.pop(key, None)

    def refresh_binding(
        self: Any,
        stage: str,
        payload: Mapping[str, Any],
        allowed_server_ids: Collection[str] | None,
    ) -> dict[str, Any]:
        key = _binding_key(
            external_agent_bridge_module,
            stage=stage,
            payload=payload,
            allowed_server_ids=allowed_server_ids,
        )
        # The core bridge caches schema results indefinitely.  A schema query is a
        # live contract boundary, so bypass any stale cache before selecting an owner.
        with self._lock:
            self._schema_cache.pop(key, None)
        result = current(
            self,
            stage,
            external_agent_bridge_module.SCHEMA_TOOL,
            payload,
            allowed_server_ids=allowed_server_ids,
        )
        if str(result.get("status", "")) != "PASS":
            invalidate(self, key)
            return result
        try:
            schema = validate_input_schema(
                result.get("input_schema"),
                owner=(
                    "external schema binding "
                    f"{result.get('server')!r}/{result.get('tool')!r}"
                ),
            )
        except Exception as exc:
            invalidate(self, key)
            raise external_agent_bridge_module.ExternalAgentBridgeError(str(exc)) from exc
        binding = {
            "server": str(result.get("server", "")).strip(),
            "tool": str(result.get("tool", "")).strip(),
            "schema_sha256": _fingerprint(schema),
            "target_args": dict(result.get("target_args_injected_by_router", {}) or {}),
            "access": str(result.get("access", "read")).strip() or "read",
        }
        if not binding["server"] or not binding["tool"]:
            invalidate(self, key)
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "Live external MCP schema did not identify an exact provider/tool owner."
            )
        with self._lock:
            _binding_store(self)[key] = binding
        return result

    def live_bound_route(
        self: Any,
        *,
        stage: str,
        capability: str,
        target: Mapping[str, str],
        max_access: str,
        allowed_server_ids: Collection[str] | None,
        binding: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        router = self._external_router()
        allowed = _server_scope(allowed_server_ids)
        routes = router.registry.routes(
            capability,
            stage=stage,
            minecraft_version=target["minecraft_version"],
            loader=target["loader"],
            max_access=max_access,
        )
        matches = [
            route
            for route in routes
            if str(route["server"]) == binding["server"]
            and str(route["route"].get("tool", "")).strip() == binding["tool"]
            and (allowed is None or str(route["server"]) in allowed)
        ]
        if len(matches) != 1:
            raise ExternalMCPSchemaBindingError(
                "Bound external MCP route disappeared or became ambiguous"
            )
        route = matches[0]
        if not router._configured(route["entry"]):
            raise ExternalMCPSchemaBindingError(
                f"Bound external MCP provider {binding['server']!r} is unavailable"
            )
        if dict(route["route"].get("target_args", {}) or {}) != dict(
            binding.get("target_args", {}) or {}
        ):
            raise ExternalMCPSchemaBindingError(
                "Bound external MCP target-argument projection changed"
            )
        return route

    @wraps(current)
    def call(
        self: Any,
        stage: str,
        name: str,
        payload: Mapping[str, Any],
        *,
        allowed_server_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        if name == external_agent_bridge_module.SCHEMA_TOOL:
            return refresh_binding(self, stage, payload, allowed_server_ids)
        if name != external_agent_bridge_module.CALL_TOOL:
            return current(
                self,
                stage,
                name,
                payload,
                allowed_server_ids=allowed_server_ids,
            )

        if stage not in external_agent_bridge_module.AGENT_STAGES:
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                f"External MCP federation is unavailable in stage {stage!r}."
            )
        target = external_agent_bridge_module._target(payload)
        max_access = str(payload.get("max_access", "read")).strip().lower() or "read"
        if max_access not in {"read", "write", "admin"}:
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "max_access must be read, write or admin"
            )
        if stage != "runtime" and max_access != "read":
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "Non-runtime external MCP access is read-only"
            )
        capability = str(payload.get("capability", "")).strip()
        if not capability:
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "capability must not be empty"
            )
        raw_arguments = payload.get("arguments", {})
        if raw_arguments is None:
            raw_arguments = {}
        if not isinstance(raw_arguments, Mapping):
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "arguments must be an object"
            )
        corroborate = payload.get("corroborate", 1)
        if type(corroborate) is not int or not 1 <= corroborate <= 4:
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "corroborate must be an integer from 1 to 4"
            )
        if corroborate != 1:
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "Schema-bound external MCP execution requires corroborate=1; "
                "cross-provider corroboration must discover and validate each provider schema "
                "explicitly instead of silently sharing one provider's contract."
            )
        disposable_runtime = payload.get("disposable_runtime", False)
        if type(disposable_runtime) is not bool:
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "disposable_runtime must be a boolean"
            )
        if (
            stage == "runtime"
            and max_access in {"write", "admin"}
            and not disposable_runtime
        ):
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "Write/admin Minecraft MCP tools require a disposable runtime instance."
            )

        allowed = _server_scope(allowed_server_ids)
        key = _binding_key(
            external_agent_bridge_module,
            stage=stage,
            payload=payload,
            allowed_server_ids=allowed,
        )
        with self._lock:
            binding = _binding_store(self).get(key)
        if binding is None:
            described = refresh_binding(
                self,
                stage,
                _schema_payload(payload),
                allowed,
            )
            if str(described.get("status", "")) != "PASS":
                raise external_agent_bridge_module.ExternalAgentBridgeError(
                    "No live external MCP schema is available for this call scope."
                )
            with self._lock:
                binding = _binding_store(self).get(key)
        if binding is None:  # pragma: no cover - defensive invariant
            raise external_agent_bridge_module.ExternalAgentBridgeError(
                "External MCP schema binding was not established"
            )

        router = self._external_router()
        try:
            route = live_bound_route(
                self,
                stage=stage,
                capability=capability,
                target=target,
                max_access=max_access,
                allowed_server_ids=allowed,
                binding=binding,
            )
            live = self._run_async(
                external_agent_bridge_module._provider_schema,
                route["entry"],
                tool=binding["tool"],
                env=router._child_env(route["entry"]),
                url=router._server_url(route["entry"]),
                timeout_seconds=min(self.timeout_seconds, 120.0),
            )
            live_schema = validate_input_schema(
                live.get("input_schema"),
                owner=(
                    "live external provider "
                    f"{binding['server']!r}/{binding['tool']!r}"
                ),
            )
            if _fingerprint(live_schema) != binding["schema_sha256"]:
                raise ExternalMCPSchemaBindingError(
                    "External MCP provider schema changed after discovery; refresh schema"
                )

            arguments = dict(raw_arguments)
            for reserved in self._reserved_target_arguments(
                router,
                capability=capability,
                stage=stage,
                target=target,
                max_access=max_access,
                allowed_server_ids=allowed,
            ):
                arguments.pop(reserved, None)
            return router.invoke_bound(
                capability,
                stage=stage,
                server=binding["server"],
                tool=binding["tool"],
                arguments=arguments,
                target=target,
                max_access=max_access,
                disposable_runtime=disposable_runtime,
                allowed_server_ids=allowed,
            )
        except Exception as exc:
            invalidate(self, key)
            if isinstance(exc, external_agent_bridge_module.ExternalAgentBridgeError):
                raise
            raise external_agent_bridge_module.ExternalAgentBridgeError(str(exc)) from exc

    setattr(call, _MARKER, True)
    call.__wrapped__ = current  # type: ignore[attr-defined]
    bridge_class.call = call


__all__ = ["ExternalMCPSchemaBindingError", "install"]
