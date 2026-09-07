from __future__ import annotations

"""Native, bounded recovery for host-selected tool arguments.

The host owns the original argument container.  When a model cannot emit the already
selected tool directly, this module decomposes its object schema into small native
function-call pages, validates each page, merges them, then validates the final object
against the original schema.  No recovery turn asks the model to author a JSON document.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_ARGUMENT_PAGE_TOOL = "mmm_submit_argument_page"
_MAX_PAGE_PROPERTIES = 4
_MAX_REPAIR_ERROR_CHARS = 1200


def _tool_schema(parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _ARGUMENT_PAGE_TOOL,
            "description": (
                "Submit exactly one bounded page of arguments for an action that the host "
                "has already selected. Do not choose another action."
            ),
            "parameters": dict(parameters),
        },
    }


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
    action_name: str,
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
    fields = ", ".join(str(name) for name in properties) if isinstance(properties, Mapping) else "arguments"
    instruction = (
        f"HOST ACTION IS FIXED: {action_name}. Supply argument page {page_index}/{page_count}. "
        f"Call the only available function exactly once. This page owns only these fields: {fields}. "
        "Do not emit JSON prose, YAML, XML, a code fence, or another function name. "
        "The host owns the complete argument object and will merge and validate pages."
    )
    if repair_error:
        instruction += (
            " The previous native argument page was invalid. Repair only this page. Validation: "
            + repair_error[:_MAX_REPAIR_ERROR_CHARS]
        )
    messages.append({"role": "user", "content": instruction})
    return tuple(messages)


def _request(
    request: Any,
    *,
    action_name: str,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    repair_error: str = "",
) -> Any:
    # Import lazily so runtime bootstrap can install the global atomicity boundary
    # without creating a module cycle.
    from .model_output_atomicity_contract import assert_atomic_model_schema

    assert_atomic_model_schema(page_schema, surface=f"native argument page for {action_name!r}")
    schema = _tool_schema(page_schema)
    return replace(
        request,
        messages=_messages(
            request,
            action_name=action_name,
            page_index=page_index,
            page_count=page_count,
            page_schema=page_schema,
            repair_error=repair_error,
        ),
        tools=(schema,),
        tool_validation_schemas=(schema,),
        tool_choice="required",
        parallel_tool_calls=False,
        response_format="text",
        response_schema=None,
    )


def _page_result(
    turn: Any,
    page_schema: Mapping[str, Any],
    *,
    forced: Any,
) -> tuple[dict[str, Any] | None, str, str]:
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    if len(calls) != 1:
        reason = f"expected one native argument-page call, received {len(calls)}"
        return None, reason, hashlib.sha256(reason.encode()).hexdigest()
    call = calls[0]
    call_name = str(getattr(call, "name", "") or "").strip()
    if call_name != _ARGUMENT_PAGE_TOOL:
        reason = f"unexpected native argument-page tool {call_name or '<empty>'!r}"
        return None, reason, hashlib.sha256(reason.encode()).hexdigest()
    arguments = getattr(call, "arguments", None)
    if not isinstance(arguments, Mapping):
        reason = "native argument-page call did not contain an argument object"
        return None, reason, hashlib.sha256(reason.encode()).hexdigest()
    normalized = dict(arguments)
    if not forced._arguments_match_schema(normalized, page_schema):
        reason = "native argument-page arguments failed the host page schema"
        fingerprint = hashlib.sha256(repr(sorted(normalized.items())).encode()).hexdigest()
        return None, reason, fingerprint
    fingerprint = hashlib.sha256(repr(sorted(normalized.items())).encode()).hexdigest()
    return normalized, "", fingerprint


def _page_attempt(
    current: Any,
    adapter: Any,
    request: Any,
    page_schema: Mapping[str, Any],
    *,
    forced: Any,
) -> tuple[dict[str, Any] | None, str, str]:
    try:
        turn = current(adapter, request)
    except BaseException as exc:
        cause = getattr(exc, "cause", exc)
        reason = f"{type(cause).__name__}: {cause}"[:_MAX_REPAIR_ERROR_CHARS]
        fingerprint = hashlib.sha256(reason.encode()).hexdigest()
        return None, reason, fingerprint
    return _page_result(turn, page_schema, forced=forced)


def host_selected_argument_turn(
    current: Any,
    adapter: Any,
    request: Any,
    name: str,
    *,
    forced: Any,
    prefix: str = "host_action",
) -> Any:
    """Recover one already-selected action through bounded native argument pages."""

    from .model_adapters import ModelConfigurationError

    parameters = forced._parameters(forced._selected_schema(request, name))
    try:
        pages = _argument_pages(parameters)
    except ValueError as exc:
        raise ModelConfigurationError(str(exc)) from exc

    merged: dict[str, Any] = {}
    for page_index, page_schema in enumerate(pages, start=1):
        first_request = _request(
            request,
            action_name=name,
            page_index=page_index,
            page_count=len(pages),
            page_schema=page_schema,
        )
        arguments, error, first_fingerprint = _page_attempt(
            current,
            adapter,
            first_request,
            page_schema,
            forced=forced,
        )
        if arguments is None:
            repair_request = _request(
                request,
                action_name=name,
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
                forced=forced,
            )
            if arguments is None:
                fixed_point = first_fingerprint == second_fingerprint
                suffix = (
                    "repeated-invalid native argument-page fixed point"
                    if fixed_point
                    else "bounded native argument-page repair exhausted"
                )
                raise ModelConfigurationError(
                    f"Host-selected action {name!r} {suffix} on page {page_index}/{len(pages)}; "
                    f"error={repair_error or error}."
                )
        overlap = set(merged).intersection(arguments)
        if overlap:
            raise ModelConfigurationError(
                f"HOST_ARGUMENT_DECOMPOSITION: duplicate fields across pages: {', '.join(sorted(overlap))}"
            )
        merged.update(arguments)

    if not forced._arguments_match_schema(merged, parameters):
        raise ModelConfigurationError(
            f"HOST_ARGUMENT_DECOMPOSITION: merged native pages for {name!r} failed the original schema"
        )
    return forced._response_for_call(name, merged, prefix=prefix)


def host_selected_mutation_turn(
    current: Any,
    adapter: Any,
    request: Any,
    name: str,
    *,
    forced: Any,
) -> Any:
    return host_selected_argument_turn(
        current,
        adapter,
        request,
        name,
        forced=forced,
        prefix="host_mutation",
    )


def install_into(forced: Any) -> None:
    """Replace both legacy raw-JSON recovery entry points with native atomic recovery."""

    def argument_turn(
        current: Any,
        adapter: Any,
        request: Any,
        name: str,
        *,
        prefix: str = "host_action",
    ) -> Any:
        return host_selected_argument_turn(
            current,
            adapter,
            request,
            name,
            forced=forced,
            prefix=prefix,
        )

    def mutation_turn(current: Any, adapter: Any, request: Any, name: str) -> Any:
        return host_selected_mutation_turn(
            current,
            adapter,
            request,
            name,
            forced=forced,
        )

    argument_turn.__module__ = __name__
    mutation_turn.__module__ = __name__
    forced.host_selected_argument_turn = argument_turn
    forced.host_selected_mutation_turn = mutation_turn


__all__ = [
    "host_selected_argument_turn",
    "host_selected_mutation_turn",
    "install_into",
]
