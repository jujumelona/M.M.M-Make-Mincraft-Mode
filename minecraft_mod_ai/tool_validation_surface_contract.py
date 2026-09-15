from __future__ import annotations

"""Own the native model-visible and host-validation tool-schema boundary.

The model-visible tool frontier and the host-authorized validation surface are
intentionally different capabilities. Hidden/stale schemas may be used to parse and
validate a tool name that already exists in the transcript, but they never become
model-visible and never grant execution authority. Execution remains guarded by the
current HostRunState phase/tool allowlist.

Native llama parsing owns validation directly. The final transport boundary additionally
reasserts the exact model-visible schemas after generic llama tuning wrappers have run,
so transport policy may change budgets/cache/sampling but cannot silently weaken,
rewrite, or widen a function schema.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import wraps
from typing import Any

_TRANSPORT_MARKER = "_mmm_exact_model_visible_tool_schema_v1"


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        to_schema = getattr(schema, "to_schema", None)
        if callable(to_schema):
            schema = to_schema()
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _normalized_schema(schema: Any) -> Mapping[str, Any]:
    if not isinstance(schema, Mapping):
        to_schema = getattr(schema, "to_schema", None)
        if callable(to_schema):
            schema = to_schema()
    if not isinstance(schema, Mapping):
        raise TypeError("tool definition must be a mapping or expose to_schema()")
    return schema


def _assert_unique_schema_names(
    schemas: Sequence[Any],
    *,
    surface: str,
) -> None:
    """Reject ambiguous same-name ownership inside one schema surface."""

    seen: set[str] = set()
    for schema in schemas:
        name = _tool_name(schema)
        if not name:
            continue
        if name in seen:
            raise RuntimeError(
                f"duplicate tool schema name {name!r} in {surface} surface"
            )
        seen.add(name)


def _validation_surface(
    visible: Sequence[Any],
    authorized: Sequence[Any],
) -> tuple[Any, ...]:
    """Merge parse-only schemas without overriding schemas shown this turn."""

    _assert_unique_schema_names(visible, surface="model-visible")
    _assert_unique_schema_names(authorized, surface="authorized-validation")
    result = list(visible)
    visible_names = {
        name
        for schema in visible
        if (name := _tool_name(schema))
    }
    for schema in authorized:
        name = _tool_name(schema)
        if name and name in visible_names:
            continue
        result.append(schema)
    return tuple(result)


def _exact_model_visible_transport(
    declared: Sequence[Any],
    transported: Sequence[Any],
) -> tuple[Mapping[str, Any], ...]:
    """Restore exact declared schemas for only the tools selected by inner transport.

    Inner transport may legitimately narrow a named ``tool_choice`` to one declared
    function. It may not invent a tool or rewrite that tool's schema. Matching by name
    preserves that narrowing while deep-copying the host-owned declaration verbatim.
    """

    _assert_unique_schema_names(declared, surface="model-visible")
    by_name: dict[str, Mapping[str, Any]] = {}
    for raw in declared:
        schema = _normalized_schema(raw)
        name = _tool_name(schema)
        if not name:
            raise RuntimeError("model-visible tool schema has no function name")
        by_name[name] = schema

    result: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for raw in transported:
        name = _tool_name(raw)
        if not name or name not in by_name:
            raise RuntimeError(
                f"llama transport exposed undeclared tool schema {name or '<unnamed>'!r}"
            )
        if name in seen:
            raise RuntimeError(f"llama transport duplicated tool schema {name!r}")
        seen.add(name)
        result.append(deepcopy(dict(by_name[name])))
    return tuple(result)


def _install_exact_model_visible_transport() -> None:
    """Make the late native transport boundary schema-preserving by construction."""

    from . import llama_server_hardware_policy

    current = llama_server_hardware_policy._server_payload
    if bool(getattr(current, _TRANSPORT_MARKER, False)):
        return

    @wraps(current)
    def payload(adapter: Any, request: Any) -> dict[str, Any]:
        result = dict(current(adapter, request))
        declared = tuple(getattr(request, "tools", ()) or ())
        if not declared:
            return result
        transported = result.get("tools")
        if not isinstance(transported, Sequence) or isinstance(
            transported, (str, bytes, bytearray)
        ):
            raise RuntimeError("native llama tool transport dropped the tools surface")
        result["tools"] = list(_exact_model_visible_transport(declared, transported))
        return result

    setattr(payload, _TRANSPORT_MARKER, True)
    llama_server_hardware_policy._server_payload = payload


def install() -> None:
    """Assert native validation ownership and finalize exact model-visible transport."""

    from .model_adapters import llama_cpp_adapter

    owner = getattr(llama_cpp_adapter, "_request_tool_schema_map", None)
    if not callable(owner) or not getattr(
        owner, "_mmm_core_validation_surface", False
    ):
        raise RuntimeError(
            "native llama adapter does not own the authorized tool validation surface"
        )
    _install_exact_model_visible_transport()


__all__ = [
    "_assert_unique_schema_names",
    "_exact_model_visible_transport",
    "_validation_surface",
    "install",
]
