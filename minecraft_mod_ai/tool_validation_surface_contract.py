from __future__ import annotations

"""Separate model-visible tool schemas from the host-authorized parse surface.

A live causal frontier may legitimately hide a tool that appeared earlier in the same
assistant transcript. Qwen can still emit that stale tool name because it is part of
its context. Parsing such a call must not grant execution authority, but it also must
not crash the backend before the host execution gate can reject it and return a normal
tool observation.

The current model-visible schema is authoritative for every tool that is visible on
this turn. The broader authorized surface contributes only names that are absent from
the visible frontier. This prevents a stale/raw host schema for the same tool name from
replacing the exact schema that was shown to the model.
"""

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .runtime_contract_wrappers import contract_wraps, has_contract_marker

_PARSE_MARKER = "_mmm_authorized_tool_validation_surface"
_CONTINUATION_MARKER = "_mmm_tool_validation_continuation"


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name", "")).strip()


def _validation_surface(
    visible: Sequence[Any],
    authorized: Sequence[Any],
) -> tuple[Any, ...]:
    """Merge parse-only schemas without overriding schemas shown this turn."""

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


def install() -> None:
    from .model_adapters import llama_cpp_adapter

    current_parse = llama_cpp_adapter._qwen_tool_generation_response
    if not has_contract_marker(current_parse, _PARSE_MARKER):

        @contract_wraps(current_parse)
        def parse_with_authorized_surface(message: Any, request: Any):
            visible = tuple(getattr(request, "tools", ()) or ())
            validation = tuple(
                getattr(request, "tool_validation_schemas", ()) or ()
            )
            if validation:
                request = replace(
                    request,
                    tools=_validation_surface(visible, validation),
                )
            return current_parse(message, request)

        setattr(parse_with_authorized_surface, _PARSE_MARKER, True)
        llama_cpp_adapter._qwen_tool_generation_response = parse_with_authorized_surface

    current_continuation = llama_cpp_adapter._reasoning_continuation_request
    if not has_contract_marker(current_continuation, _CONTINUATION_MARKER):

        @contract_wraps(current_continuation)
        def continuation_with_authorized_surface(request: Any, reasoning: str):
            continued = current_continuation(request, reasoning)
            validation = tuple(
                getattr(request, "tool_validation_schemas", ()) or ()
            )
            if not validation:
                return continued
            return replace(continued, tool_validation_schemas=validation)

        setattr(continuation_with_authorized_surface, _CONTINUATION_MARKER, True)
        llama_cpp_adapter._reasoning_continuation_request = (
            continuation_with_authorized_surface
        )


__all__ = ["_validation_surface", "install"]
