from __future__ import annotations

"""Runtime safety for llama.cpp tool schemas.

Pre-design research is now owned directly by ``pre_design_grounded_rag`` and
``small_model_predesign_research``. This runtime contract must not import, wrap, or
mutate a pre-design research owner.
"""

import copy
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

_INSTALLED = False


def _resolve_local_ref(
    schema: Mapping[str, Any], root: Mapping[str, Any]
) -> Mapping[str, Any]:
    ref = schema.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return schema
    value: Any = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or token not in value:
            return schema
        value = value[token]
    return value if isinstance(value, Mapping) else schema


def _inferred_scalar_type(values: Sequence[Any]) -> str | None:
    if not values:
        return None
    if all(isinstance(value, bool) for value in values):
        return "boolean"
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return "integer"
    if all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in values
    ):
        return "number"
    if all(isinstance(value, str) for value in values):
        return "string"
    return None


def _grammar_safe_schema(
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project JSON Schema to llama.cpp's conservative grammar subset."""

    root = schema if root is None else root
    resolved = _resolve_local_ref(schema, root)
    if resolved is not schema:
        return _grammar_safe_schema(resolved, root=root)

    for union_key in ("anyOf", "oneOf", "allOf"):
        branches = resolved.get(union_key)
        if isinstance(branches, list):
            candidates = [item for item in branches if isinstance(item, Mapping)]
            non_null = [item for item in candidates if item.get("type") != "null"]
            if non_null:
                return _grammar_safe_schema(non_null[0], root=root)

    raw_type = resolved.get("type")
    if isinstance(raw_type, list):
        types = [str(item) for item in raw_type if str(item) != "null"]
        raw_type = types[0] if types else "string"
    if not isinstance(raw_type, str):
        if isinstance(resolved.get("properties"), Mapping):
            raw_type = "object"
        elif "items" in resolved:
            raw_type = "array"
        else:
            enum = resolved.get("enum")
            raw_type = (
                _inferred_scalar_type(enum) if isinstance(enum, list) else None
            ) or "string"

    if raw_type == "object":
        properties_raw = resolved.get("properties")
        properties: dict[str, Any] = {}
        if isinstance(properties_raw, Mapping):
            for key, value in properties_raw.items():
                properties[str(key)] = (
                    _grammar_safe_schema(value, root=root)
                    if isinstance(value, Mapping)
                    else {"type": "string"}
                )
        result: dict[str, Any] = {"type": "object", "properties": properties}
        required_raw = resolved.get("required")
        if isinstance(required_raw, list):
            required = [str(key) for key in required_raw if str(key) in properties]
            if required:
                result["required"] = required
        return result

    if raw_type == "array":
        items = resolved.get("items")
        return {
            "type": "array",
            "items": (
                _grammar_safe_schema(items, root=root)
                if isinstance(items, Mapping)
                else {"type": "string"}
            ),
        }

    if raw_type not in {"string", "integer", "number", "boolean", "null"}:
        raw_type = "string"
    result = {"type": raw_type}
    enum = resolved.get("enum")
    if isinstance(enum, list) and enum and _inferred_scalar_type(enum) == raw_type:
        result["enum"] = list(enum)
    return result


def _grammar_safe_tool(tool: Mapping[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, Mapping):
        return copy.deepcopy(dict(tool))
    safe_function: dict[str, Any] = {"name": str(function.get("name", ""))}
    description = function.get("description")
    if isinstance(description, str) and description:
        safe_function["description"] = description
    parameters = function.get("parameters")
    if isinstance(parameters, Mapping):
        projected = _grammar_safe_schema(parameters)
        if projected.get("type") != "object":
            projected = {"type": "object", "properties": {}}
    else:
        projected = {"type": "object", "properties": {}}
    safe_function["parameters"] = projected
    return {"type": "function", "function": safe_function}


def _install_llama_tool_schema_projection(policy_module: Any) -> None:
    current_payload = policy_module._server_payload
    if getattr(current_payload, "_mmm_grammar_safe_tools_v1", False):
        return

    @wraps(current_payload)
    def server_payload(adapter: Any, request: Any) -> dict[str, Any]:
        payload = current_payload(adapter, request)
        tools = payload.get("tools")
        if isinstance(tools, list):
            payload = dict(payload)
            payload["tools"] = [
                _grammar_safe_tool(tool)
                for tool in tools
                if isinstance(tool, Mapping)
            ]
        return payload

    server_payload._mmm_grammar_safe_tools_v1 = True
    policy_module._server_payload = server_payload


def install() -> None:
    """Install only the llama tool-schema boundary; pre-design has no runtime patch."""

    global _INSTALLED
    if _INSTALLED:
        return
    from . import llama_server_hardware_policy

    _install_llama_tool_schema_projection(llama_server_hardware_policy)
    _INSTALLED = True


__all__ = [
    "_grammar_safe_schema",
    "_grammar_safe_tool",
    "_install_llama_tool_schema_projection",
    "install",
]
