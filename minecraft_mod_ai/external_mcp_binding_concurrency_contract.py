from __future__ import annotations

"""Serialize one external MCP schema owner per exact authorization scope.

The schema/provider binding contract prevents one call from failing over after schema
validation, but the bridge is shared by concurrent agents. Two schema requests for the
same scope could therefore race and let the later request replace the binding observed
by the earlier request. This layer keeps same-scope schema/call operations serialized
and reuses the first live reviewed owner until that owner drifts or becomes unavailable.
"""

import copy
import threading
from collections.abc import Collection, Mapping
from functools import wraps
from typing import Any

from . import external_mcp_binding_contract as binding_contract
from .mcp_schema_integrity_contract import validate_input_schema

_MARKER = "_mmm_external_mcp_scope_serialization_v1"
_RESPONSES_ATTR = "_mmm_external_schema_bound_responses"
_LOCK_COUNT = 32
_SCOPE_LOCKS = tuple(threading.RLock() for _ in range(_LOCK_COUNT))


def _scope_lock(key: object) -> threading.RLock:
    return _SCOPE_LOCKS[hash(key) % _LOCK_COUNT]


def _response_store(bridge: Any) -> dict[Any, dict[str, Any]]:
    store = getattr(bridge, _RESPONSES_ATTR, None)
    if store is None:
        store = {}
        setattr(bridge, _RESPONSES_ATTR, store)
    return store


def _invalidate(bridge: Any, key: object) -> None:
    with bridge._lock:
        binding_contract._binding_store(bridge).pop(key, None)
        _response_store(bridge).pop(key, None)
        bridge._schema_cache.pop(key, None)


def _revalidate_existing(
    bridge: Any,
    bridge_module: Any,
    *,
    stage: str,
    payload: Mapping[str, Any],
    allowed_server_ids: Collection[str] | None,
    binding: Mapping[str, Any],
) -> None:
    target = bridge_module._target(payload)
    max_access = str(payload.get("max_access", "read")).strip().lower() or "read"
    capability = str(payload.get("capability", "")).strip()
    router = bridge._external_router()
    route = binding_contract._resolve_exact_route(
        router,
        capability=capability,
        stage=stage,
        target=target,
        max_access=max_access,
        allowed_server_ids=allowed_server_ids,
        server=str(binding["server"]),
        tool=str(binding["tool"]),
    )
    route_spec = route["route"]
    live_access = str(route_spec.get("access", "read")).strip() or "read"
    if live_access != str(binding.get("access", "read")):
        raise binding_contract.ExternalMCPSchemaBindingError(
            "Bound external MCP access changed after schema discovery"
        )
    if dict(route_spec.get("target_args", {}) or {}) != dict(
        binding.get("target_args", {}) or {}
    ):
        raise binding_contract.ExternalMCPSchemaBindingError(
            "Bound external MCP target-argument projection changed"
        )
    if binding_contract._route_fingerprint(router, route) != str(
        binding.get("route_sha256", "")
    ):
        raise binding_contract.ExternalMCPSchemaBindingError(
            "Bound external MCP provider/route identity changed after schema discovery"
        )
    live = binding_contract._run_live_provider_schema(
        bridge,
        bridge_module,
        router,
        route,
        binding,
    )
    live_schema = validate_input_schema(
        live.get("input_schema"),
        owner=(
            "live external provider "
            f"{binding['server']!r}/{binding['tool']!r}"
        ),
    )
    if binding_contract._fingerprint(live_schema) != str(binding["schema_sha256"]):
        raise binding_contract.ExternalMCPSchemaBindingError(
            "External MCP provider schema changed after discovery; refresh schema"
        )


def install(external_agent_bridge_module: Any) -> None:
    """Keep concurrent same-scope schema discovery and execution on one live owner."""

    bridge_class = external_agent_bridge_module.ExternalAgentBridge
    current = bridge_class.call
    if bool(getattr(current, _MARKER, False)):
        return
    if not bool(getattr(current, "_mmm_external_mcp_schema_binding_v1", False)):
        raise RuntimeError(
            "external MCP binding concurrency requires schema/provider binding first"
        )

    @wraps(current)
    def call(
        self: Any,
        stage: str,
        name: str,
        payload: Mapping[str, Any],
        *,
        allowed_server_ids: Collection[str] | None = None,
    ) -> dict[str, Any]:
        if name not in {
            external_agent_bridge_module.SCHEMA_TOOL,
            external_agent_bridge_module.CALL_TOOL,
        }:
            return current(
                self,
                stage,
                name,
                payload,
                allowed_server_ids=allowed_server_ids,
            )

        try:
            key = binding_contract._binding_key(
                external_agent_bridge_module,
                stage=stage,
                payload=payload,
                allowed_server_ids=allowed_server_ids,
            )
        except Exception:
            # Preserve the bridge's canonical validation/error behavior for malformed
            # requests that cannot yet form an authorization scope key.
            return current(
                self,
                stage,
                name,
                payload,
                allowed_server_ids=allowed_server_ids,
            )

        with _scope_lock(key):
            if name == external_agent_bridge_module.SCHEMA_TOOL:
                with self._lock:
                    binding = binding_contract._binding_store(self).get(key)
                    response = _response_store(self).get(key)
                if binding is not None and response is not None:
                    try:
                        _revalidate_existing(
                            self,
                            external_agent_bridge_module,
                            stage=stage,
                            payload=payload,
                            allowed_server_ids=allowed_server_ids,
                            binding=binding,
                        )
                    except Exception as exc:
                        _invalidate(self, key)
                        if isinstance(
                            exc, external_agent_bridge_module.ExternalAgentBridgeError
                        ):
                            raise
                        raise external_agent_bridge_module.ExternalAgentBridgeError(
                            str(exc)
                        ) from exc
                    return copy.deepcopy(response)

                result = current(
                    self,
                    stage,
                    name,
                    payload,
                    allowed_server_ids=allowed_server_ids,
                )
                if str(result.get("status", "")) == "PASS":
                    with self._lock:
                        binding = binding_contract._binding_store(self).get(key)
                        if binding is not None:
                            _response_store(self)[key] = copy.deepcopy(result)
                return result

            try:
                return current(
                    self,
                    stage,
                    name,
                    payload,
                    allowed_server_ids=allowed_server_ids,
                )
            except Exception:
                # The inner binding layer invalidates on provider/schema drift. Keep the
                # cached model-facing response in lockstep with that binding.
                with self._lock:
                    if key not in binding_contract._binding_store(self):
                        _response_store(self).pop(key, None)
                raise

    setattr(call, _MARKER, True)
    call.__wrapped__ = current  # type: ignore[attr-defined]
    bridge_class.call = call


def assert_installed(external_agent_bridge_module: Any) -> None:
    target = external_agent_bridge_module.ExternalAgentBridge.call
    if getattr(target, _MARKER, False) is not True:
        raise RuntimeError("external MCP same-scope binding serialization is not installed")


__all__ = ["assert_installed", "install"]
