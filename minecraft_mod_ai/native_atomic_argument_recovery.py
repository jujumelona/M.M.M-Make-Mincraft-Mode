from __future__ import annotations

"""Bounded argument-only recovery for a host-selected action.

The host has already selected the semantic action before this module runs. Recovery must
therefore not ask a small model to select a tool again. Each bounded page is requested as
a JSON object constrained by the page's argument schema, model-returned tool calls are
ignored, and the host constructs the final ToolCall only after every page validates.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_MAX_PAGE_PROPERTIES = 4
_MAX_REPAIR_ERROR_CHARS = 1200


def _forced_module() -> Any:
    # Import lazily to avoid a module cycle while forced_tool_execution_contract imports
    # the public recovery entry points on demand.
    from . import forced_tool_execution_contract

    return forced_tool_execution_contract


def _required_names(schema: Mapping[str, Any]) -> tuple[str, ...]:
    raw = schema.get("required", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(str(value) for value in raw)


def _page_schema(
    source: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    properties = source.get("properties")
    if not isinstance(properties, Mapping):
        return dict(source)
    required = set(_required_names(source))
    page_properties = {
        name: dict(properties[name]) if isinstance(properties[name], Mapping) else properties[name]
        for name in names
    }
    page: dict[str, Any] = {
        "type": "object",
        "properties": page_properties,
        "additionalProperties": False,
    }
    page_required = [name for name in names if name in required]
    if page_required:
        page["required"] = page_required
    for keyword in ("$defs", "definitions"):
        definitions = source.get(keyword)
        if isinstance(definitions, Mapping):
            page[keyword] = dict(definitions)
    return page


def _argument_pages(parameters: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    properties = parameters.get("properties")
    if parameters.get("type") != "object" or not isinstance(properties, Mapping):
        raise ValueError(
            "HOST_ARGUMENT_DECOMPOSITION: function parameters must be an object schema"
        )
    names = tuple(str(name) for name in properties)
    if not names:
        return (dict(parameters),)
    return tuple(
        _page_schema(parameters, names[index : index + _MAX_PAGE_PROPERTIES])
        for index in range(0, len(names), _MAX_PAGE_PROPERTIES)
    )


def _messages(
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    repair_error: str = "",
) -> tuple[dict[str, Any], ...]:
    messages = [
        dict(raw)
        for raw in tuple(getattr(request, "messages", ()) or ())
        if isinstance(raw, Mapping)
    ]
    properties = page_schema.get("properties")
    fields = (
        ", ".join(str(name) for name in properties)
        if isinstance(properties, Mapping)
        else "arguments"
    )
    schema_hint = json.dumps(
        dict(page_schema),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    instruction = (
        f"The host already selected the action. Supply argument page {page_index}/{page_count} "
        f"as one JSON object containing only these fields: {fields}. "
        f"Exact page JSON schema: {schema_hint}. "
        "Do not choose or name a tool. Do not emit prose, YAML, XML, or a code fence. "
        "The host owns action selection, merges bounded pages, validates the complete object, "
        "and constructs the final tool call."
    )
    if repair_error:
        instruction += (
            " Repair the arguments only. The previous argument object was invalid. Validation: "
            + repair_error[:_MAX_REPAIR_ERROR_CHARS]
        )
    messages.append({"role": "user", "content": instruction})
    return tuple(messages)


def _request(
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    repair_error: str = "",
) -> Any:
    # The boundary is checked per model-authored page, never against the host-owned
    # original container. A single oversized nested field therefore fails closed before
    # generation rather than falling back to an unconstrained response.
    from .model_output_atomicity_contract import assert_atomic_model_schema

    assert_atomic_model_schema(
        page_schema,
        surface="host-selected argument page",
    )
    return replace(
        request,
        messages=_messages(
            request,
            page_index=page_index,
            page_count=page_count,
            page_schema=page_schema,
            repair_error=repair_error,
        ),
        tools=(),
        tool_validation_schemas=(),
        tool_choice=None,
        parallel_tool_calls=False,
        response_format="json",
        response_schema=dict(page_schema),
    )


def _fingerprint(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")
    return hashlib.sha256(encoded).hexdigest()


def _page_result(
    turn: Any,
    page_schema: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, str]:
    """Parse only JSON content; stale/model-authored ToolCalls never become executable."""

    forced = _forced_module()
    raw = str(getattr(turn, "content", "") or "").strip()
    if not raw:
        reason = "argument page returned no JSON object"
        return None, reason, _fingerprint({"content": raw})
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        reason = f"argument page was not valid JSON: {exc}"
        return None, reason, _fingerprint({"content": raw})
    if not isinstance(parsed, Mapping):
        reason = "argument page JSON must be an object"
        return None, reason, _fingerprint(parsed)
    normalized = dict(parsed)
    if not forced._arguments_match_schema(normalized, page_schema):
        diag = getattr(forced, "_schema_validation_diagnostics", lambda *args: "")(normalized, page_schema)
        reason = f"argument page JSON failed the host page schema ({diag})" if diag else "argument page JSON failed the host page schema"
        return None, reason, _fingerprint(normalized)
    return normalized, "", _fingerprint(normalized)


def _page_attempt(
    current: Any,
    adapter: Any,
    request: Any,
    page_schema: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, str]:
    try:
        turn = current(adapter, request)
    except BaseException as exc:
        cause = getattr(exc, "cause", exc)
        reason = f"{type(cause).__name__}: {cause}"[:_MAX_REPAIR_ERROR_CHARS]
        return None, reason, _fingerprint({"exception": reason})
    return _page_result(turn, page_schema)


def host_selected_argument_turn(
    current: Any,
    adapter: Any,
    request: Any,
    name: str,
    *,
    prefix: str = "host_action",
) -> Any:
    """Recover one already-selected action through bounded argument-only JSON pages."""

    from .model_adapters import ModelConfigurationError

    forced = _forced_module()
    parameters = forced._parameters(forced._selected_schema(request, name))
    try:
        pages = _argument_pages(parameters)
    except ValueError as exc:
        raise ModelConfigurationError(str(exc)) from exc

    merged: dict[str, Any] = {}
    for page_index, page_schema in enumerate(pages, start=1):
        first_request = _request(
            request,
            page_index=page_index,
            page_count=len(pages),
            page_schema=page_schema,
        )
        arguments, error, first_fingerprint = _page_attempt(
            current,
            adapter,
            first_request,
            page_schema,
        )
        if arguments is None:
            repair_request = _request(
                request,
                page_index=page_index,
                page_count=len(pages),
                page_schema=page_schema,
                repair_error=error,
            )
            arguments, repair_error, second_fingerprint = _page_attempt(
                current,
                adapter,
                repair_request,
                page_schema,
            )
            if arguments is None:
                fixed_point = first_fingerprint == second_fingerprint
                suffix = (
                    "repeated-invalid argument-page fixed point"
                    if fixed_point
                    else "bounded argument-page repair exhausted"
                )
                raise ModelConfigurationError(
                    f"Host-selected action {name!r} {suffix} on page "
                    f"{page_index}/{len(pages)}; error={repair_error or error}."
                )
        overlap = set(merged).intersection(arguments)
        if overlap:
            raise ModelConfigurationError(
                "HOST_ARGUMENT_DECOMPOSITION: duplicate fields across pages: "
                + ", ".join(sorted(overlap))
            )
        merged.update(arguments)

    if not forced._arguments_match_schema(merged, parameters):
        raise ModelConfigurationError(
            f"HOST_ARGUMENT_DECOMPOSITION: merged argument pages for {name!r} "
            "failed the original schema"
        )
    return forced._response_for_call(name, merged, prefix=prefix)


def host_selected_mutation_turn(
    current: Any,
    adapter: Any,
    request: Any,
    name: str,
) -> Any:
    return host_selected_argument_turn(
        current,
        adapter,
        request,
        name,
        prefix="host_mutation",
    )


__all__ = [
    "host_selected_argument_turn",
    "host_selected_mutation_turn",
]
