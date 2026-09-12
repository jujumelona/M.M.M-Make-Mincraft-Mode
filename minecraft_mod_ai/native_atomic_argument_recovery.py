from __future__ import annotations

"""Bounded argument-only recovery for a host-selected action.

The host has already selected the semantic action before this module runs. Recovery must
therefore not ask a small model to select a semantic action again. Each bounded page is
exposed as exactly one forced native function whose parameters are the page schema. The
host consumes only the returned ToolCall.arguments mapping, merges validated pages, and
constructs the final ToolCall. Model-authored JSON text is never parsed on this path.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

_MAX_PAGE_PROPERTIES = 3
_MAX_REPAIR_ERROR_CHARS = 1200
_MAX_ATOMIC_STRING_LENGTH = 256
_SOURCE_EDIT_TOOL = "apply_source_edit"


def _requires_stream_transport(schema: Mapping[str, Any]) -> bool:
    """Return true when bounding one model page would narrow the original string contract."""

    if schema.get("type") != "string" or "enum" in schema:
        return False
    max_length = schema.get("maxLength")
    return not isinstance(max_length, int) or max_length > _MAX_ATOMIC_STRING_LENGTH


_SOURCE_EDIT_OPERATION_ALIASES = {
    "create": "create_file",
    "replace": "replace_exact",
    "delete": "delete_file",
}
_SOURCE_EDIT_OPERATION_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "replace_exact": (("path", "old", "new"), ("path", "old", "new")),
    "insert_before": (("path", "anchor", "content"), ("path", "anchor", "content")),
    "insert_after": (("path", "anchor", "content"), ("path", "anchor", "content")),
    "create_file": (("path", "content"), ("path", "content")),
    "delete_file": (("path",), ("path",)),
    "create_java_type": (
        ("path", "package_name", "declaration"),
        ("path", "package_name", "declaration"),
    ),
    "add_java_import": (("path", "import_name"), ("path", "import_name")),
    "insert_java_member": (("path", "member"), ("path", "member")),
}


def _forced_module() -> Any:
    from . import forced_tool_execution_contract

    return forced_tool_execution_contract


def _required_names(schema: Mapping[str, Any]) -> tuple[str, ...]:
    raw = schema.get("required", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(str(value) for value in raw)


def _bounded_property(schema: Mapping[str, Any]) -> dict[str, Any]:
    copy = dict(schema)
    if copy.get("type") == "string" and "enum" not in copy and "maxLength" not in copy:
        copy["maxLength"] = _MAX_ATOMIC_STRING_LENGTH
    elif copy.get("type") == "array":
        if "maxItems" not in copy:
            copy["maxItems"] = 4
        if "items" in copy and isinstance(copy["items"], Mapping):
            copy["items"] = _bounded_property(copy["items"])
    return copy


def _page_schema(source: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    properties = source.get("properties")
    if not isinstance(properties, Mapping):
        return dict(source)
    required = set(_required_names(source))
    page_properties = {
        name: _bounded_property(properties[name])
        if isinstance(properties[name], Mapping)
        else properties[name]
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


def _source_edit_selector_schema(parameters: Mapping[str, Any]) -> dict[str, Any]:
    properties = parameters.get("properties")
    if parameters.get("type") != "object" or not isinstance(properties, Mapping):
        raise ValueError(
            "HOST_ARGUMENT_DECOMPOSITION: apply_source_edit parameters must be an object schema"
        )
    operation_schema = properties.get("operation")
    if not isinstance(operation_schema, Mapping):
        raise ValueError(
            "HOST_ARGUMENT_DECOMPOSITION: apply_source_edit schema is missing operation"
        )
    page: dict[str, Any] = {
        "type": "object",
        "properties": {"operation": dict(operation_schema)},
        "required": ["operation"],
        "additionalProperties": False,
    }
    for keyword in ("$defs", "definitions"):
        definitions = parameters.get(keyword)
        if isinstance(definitions, Mapping):
            page[keyword] = dict(definitions)
    return page


def _canonical_source_edit_operation(value: Any) -> str:
    operation = str(value or "").strip()
    return _SOURCE_EDIT_OPERATION_ALIASES.get(operation, operation)


def _source_edit_detail_schema(
    parameters: Mapping[str, Any],
    operation: str,
) -> dict[str, Any]:
    contract = _SOURCE_EDIT_OPERATION_FIELDS.get(operation)
    if contract is None:
        raise ValueError(
            f"HOST_ARGUMENT_DECOMPOSITION: unsupported apply_source_edit operation {operation!r}"
        )
    properties = parameters.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError(
            "HOST_ARGUMENT_DECOMPOSITION: apply_source_edit parameters must expose properties"
        )
    names, required = contract
    missing = [name for name in names if name not in properties]
    if missing:
        raise ValueError(
            "HOST_ARGUMENT_DECOMPOSITION: apply_source_edit schema is missing canonical fields: "
            + ", ".join(missing)
        )
    page: dict[str, Any] = {
        "type": "object",
        "properties": {
            name: dict(properties[name])
            if isinstance(properties[name], Mapping)
            else properties[name]
            for name in names
        },
        "required": list(required),
        "additionalProperties": False,
    }
    for keyword in ("$defs", "definitions"):
        definitions = parameters.get(keyword)
        if isinstance(definitions, Mapping):
            page[keyword] = dict(definitions)
    return page


def _source_edit_scalar_schema(detail_schema: Mapping[str, Any]) -> dict[str, Any] | None:
    properties = detail_schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    names = tuple(
        str(name)
        for name, raw_schema in properties.items()
        if not (
            isinstance(raw_schema, Mapping)
            and _requires_stream_transport(raw_schema)
        )
    )
    if not names:
        return None
    return _page_schema(detail_schema, names)


def _source_edit_stream_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "chunk": {"type": "string", "maxLength": _MAX_ATOMIC_STRING_LENGTH},
            "done": {"type": "boolean"},
        },
        "required": ["chunk", "done"],
        "additionalProperties": False,
    }


def _base_messages(request: Any, *, isolated: bool) -> list[dict[str, Any]]:
    source = [
        dict(raw)
        for raw in tuple(getattr(request, "messages", ()) or ())
        if isinstance(raw, Mapping)
    ]
    if not isolated:
        return source

    # A repeated invalid fingerprint means the current conversation state has become a
    # deterministic attractor. Crossing that boundary is a state transition, not another
    # retry: retain policy plus the latest task-bearing user turn and discard assistant/tool
    # residue that can keep reproducing the invalid prose/tool shape.
    policy = [
        message
        for message in source
        if str(message.get("role", "")).strip().casefold() in {"system", "developer"}
    ]
    latest_user = next(
        (
            message
            for message in reversed(source)
            if str(message.get("role", "")).strip().casefold() == "user"
        ),
        None,
    )
    if latest_user is not None:
        policy.append(latest_user)
    return policy


def _messages(
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    action_name: str,
    repair_error: str = "",
    context_instruction: str = "",
    isolated: bool = False,
) -> tuple[dict[str, Any], ...]:
    messages = _base_messages(request, isolated=isolated)
    properties = page_schema.get("properties")
    fields = (
        ", ".join(str(name) for name in properties)
        if isinstance(properties, Mapping)
        else "arguments"
    )
    instruction = (
        f"The host already selected action {action_name!r}. "
        f"Call that required function exactly once for argument page {page_index}/{page_count}. "
        f"Fill only these argument fields: {fields}. "
        "Do not answer in prose and do not serialize a JSON object into message content. "
        "The host owns action selection, merges bounded pages, validates the complete object, "
        "and constructs the final executable tool call."
    )
    if isolated:
        instruction += (
            " This is isolated fixed-point recovery. Ignore prior assistant/tool output; "
            "derive arguments only from the retained task context and the forced schema."
        )
    if context_instruction:
        instruction += " " + context_instruction
    if repair_error:
        instruction += (
            " Repair the function arguments only. The previous forced-call arguments were invalid. "
            "Validation: " + repair_error[:_MAX_REPAIR_ERROR_CHARS]
        )
    messages.append({"role": "user", "content": instruction})
    return tuple(messages)


def _request(
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    action_name: str,
    repair_error: str = "",
    context_instruction: str = "",
    isolated: bool = False,
) -> Any:
    from .model_output_atomicity_contract import assert_atomic_model_schema

    assert_atomic_model_schema(
        page_schema,
        surface="host-selected forced-function argument page",
    )
    page_tool = {
        "type": "function",
        "function": {
            "name": action_name,
            "description": (
                f"Fill host-selected argument page {page_index}/{page_count}. "
                "Return values only through function arguments."
            ),
            "parameters": dict(page_schema),
        },
    }
    return replace(
        request,
        messages=_messages(
            request,
            page_index=page_index,
            page_count=page_count,
            page_schema=page_schema,
            action_name=action_name,
            repair_error=repair_error,
            context_instruction=context_instruction,
            isolated=isolated,
        ),
        tools=(page_tool,),
        tool_validation_schemas=(page_tool,),
        tool_choice={"type": "function", "function": {"name": action_name}},
        parallel_tool_calls=False,
        response_format="text",
        response_schema=None,
    )


def _fingerprint(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except Exception:
        encoded = str(value)
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()


def _page_owned_arguments(
    arguments: Mapping[str, Any],
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    page_properties = page_schema.get("properties")
    all_properties = parameters.get("properties")
    if not isinstance(page_properties, Mapping) or not isinstance(all_properties, Mapping):
        return dict(arguments)
    return {
        str(name): value
        for name, value in arguments.items()
        if name in page_properties or name not in all_properties
    }


def _source_edit_stream_length_overshoot_is_safe(
    arguments: Mapping[str, Any],
    page_schema: Mapping[str, Any],
    action_name: str,
    forced: Any,
) -> bool:
    if action_name != _SOURCE_EDIT_TOOL:
        return False
    properties = page_schema.get("properties")
    if not isinstance(properties, Mapping) or set(properties) != {"chunk", "done"}:
        return False
    chunk_schema = properties.get("chunk")
    if not isinstance(chunk_schema, Mapping) or chunk_schema.get("type") != "string":
        return False
    max_length = chunk_schema.get("maxLength")
    chunk = arguments.get("chunk")
    if (
        not isinstance(max_length, int)
        or not isinstance(chunk, str)
        or len(chunk) <= max_length
    ):
        return False

    relaxed = dict(page_schema)
    relaxed_properties = dict(properties)
    relaxed_chunk = dict(chunk_schema)
    relaxed_chunk.pop("maxLength", None)
    relaxed_properties["chunk"] = relaxed_chunk
    relaxed["properties"] = relaxed_properties
    return forced._arguments_match_schema(arguments, relaxed)


def _page_result(
    turn: Any,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
    action_name: str = "submit_action",
) -> tuple[dict[str, Any] | None, str, str]:
    forced = _forced_module()
    calls = tuple(getattr(turn, "tool_calls", ()) or ())
    matches = tuple(
        call
        for call in calls
        if not action_name or str(getattr(call, "name", "")) == action_name
    )
    if len(calls) != 1 or len(matches) != 1:
        reason = (
            f"argument page must return exactly one forced {action_name!r} tool call; "
            f"received {len(calls)} tool call(s)"
        )
        return None, reason, _fingerprint(
            {
                "tool_calls": [
                    {
                        "name": str(getattr(call, "name", "")),
                        "arguments": getattr(call, "arguments", None),
                    }
                    for call in calls
                ]
            }
        )
    raw_arguments = getattr(matches[0], "arguments", None)
    if not isinstance(raw_arguments, Mapping):
        reason = "forced argument page tool call did not expose an arguments object"
        return None, reason, _fingerprint({"arguments": raw_arguments})

    normalized = _page_owned_arguments(dict(raw_arguments), page_schema, parameters)
    if not forced._arguments_match_schema(normalized, page_schema) and not (
        _source_edit_stream_length_overshoot_is_safe(
            normalized,
            page_schema,
            action_name,
            forced,
        )
    ):
        diag = getattr(forced, "_schema_validation_diagnostics", lambda *args: "")(
            normalized, page_schema
        )
        reason = (
            f"forced argument page failed the host page schema ({diag})"
            if diag
            else "forced argument page failed the host page schema"
        )
        return None, reason, _fingerprint(normalized)
    return normalized, "", _fingerprint(normalized)


def _page_attempt(
    current: Any,
    adapter: Any,
    request: Any,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
    action_name: str = "submit_action",
) -> tuple[dict[str, Any] | None, str, str]:
    try:
        turn = current(adapter, request)
    except Exception as exc:
        from .llama_finish_reason_contract import completion_boundary_error
        from .generation_output_budget import GenerationOutputBudgetError

        if completion_boundary_error(exc) is not None or isinstance(
            exc, GenerationOutputBudgetError
        ):
            raise
        cause = getattr(exc, "cause", exc)
        reason = f"{type(cause).__name__}: {cause}"[:_MAX_REPAIR_ERROR_CHARS]
        return None, reason, _fingerprint({"exception": reason})
    return _page_result(turn, page_schema, parameters, action_name)


def _recover_page(
    current: Any,
    adapter: Any,
    request: Any,
    *,
    page_index: int,
    page_count: int,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
    action_name: str,
    context_instruction: str = "",
) -> dict[str, Any]:
    from .model_adapters import ModelConfigurationError

    first_request = _request(
        request,
        page_index=page_index,
        page_count=page_count,
        page_schema=page_schema,
        action_name=action_name,
        context_instruction=context_instruction,
    )
    arguments, error, first_fingerprint = _page_attempt(
        current, adapter, first_request, page_schema, parameters, action_name
    )
    if arguments is not None:
        return arguments

    repair_request = _request(
        request,
        page_index=page_index,
        page_count=page_count,
        page_schema=page_schema,
        action_name=action_name,
        repair_error=error,
        context_instruction=context_instruction,
    )
    arguments, repair_error, second_fingerprint = _page_attempt(
        current, adapter, repair_request, page_schema, parameters, action_name
    )
    if arguments is not None:
        return arguments

    fixed_point = first_fingerprint == second_fingerprint
    if fixed_point and action_name != _SOURCE_EDIT_TOOL:
        isolated_request = _request(
            request,
            page_index=page_index,
            page_count=page_count,
            page_schema=page_schema,
            action_name=action_name,
            repair_error=repair_error or error,
            context_instruction=context_instruction,
            isolated=True,
        )
        arguments, isolated_error, _ = _page_attempt(
            current,
            adapter,
            isolated_request,
            page_schema,
            parameters,
            action_name,
        )
        if arguments is not None:
            return arguments
        raise ModelConfigurationError(
            f"Host-selected action {action_name!r} isolated forced argument-page recovery "
            f"failed after repeated-invalid fixed point on page {page_index}/{page_count}; "
            f"error={isolated_error or repair_error or error}."
        )

    if fixed_point:
        raise ModelConfigurationError(
            f"Host-selected action {action_name!r} repeated-invalid forced argument-page "
            f"fixed point on page {page_index}/{page_count}; error={repair_error or error}."
        )

    raise ModelConfigurationError(
        f"Host-selected action {action_name!r} bounded forced argument-page repair exhausted "
        f"on page {page_index}/{page_count}; error={repair_error or error}."
    )


def _recover_source_edit_stream_field(
    current: Any,
    adapter: Any,
    request: Any,
    *,
    operation: str,
    field_name: str,
    field_schema: Mapping[str, Any],
) -> str:
    from .model_adapters import ModelConfigurationError

    if field_schema.get("type") != "string":
        raise ModelConfigurationError(
            "HOST_ARGUMENT_DECOMPOSITION: streamed apply_source_edit field "
            f"{field_name!r} must be a string schema"
        )

    chunk_schema = _source_edit_stream_schema()
    accepted = ""
    while True:
        tail = accepted[-_MAX_ATOMIC_STRING_LENGTH:]
        tail_text = json.dumps(tail, ensure_ascii=False)
        instruction = (
            f"Recover apply_source_edit operation {operation!r} field {field_name!r}. "
            "Return the next exact source-text segment starting at character offset "
            f"{len(accepted)}. "
            f"The chunk must contain at most {_MAX_ATOMIC_STRING_LENGTH} characters. "
            "Set done=true only when this chunk reaches the end of this field. "
            "Continue from the stated offset; do not restart from the beginning of the field. "
            f"Accepted trailing context is {tail_text}."
        )
        piece = _recover_page(
            current,
            adapter,
            request,
            page_index=1,
            page_count=1,
            page_schema=chunk_schema,
            parameters=chunk_schema,
            action_name=_SOURCE_EDIT_TOOL,
            context_instruction=instruction,
        )
        chunk = piece.get("chunk")
        done = piece.get("done")
        if not isinstance(chunk, str) or not isinstance(done, bool):
            raise ModelConfigurationError(
                "HOST_ARGUMENT_DECOMPOSITION: source-edit stream page returned invalid chunk state"
            )
        if not done and not chunk:
            raise ModelConfigurationError(
                "HOST_ARGUMENT_DECOMPOSITION: source-edit stream made no progress "
                f"for field {field_name!r} at offset {len(accepted)}"
            )
        accepted += chunk

        max_length = field_schema.get("maxLength")
        if isinstance(max_length, int) and len(accepted) > max_length:
            raise ModelConfigurationError(
                "HOST_ARGUMENT_DECOMPOSITION: source-edit stream exceeded the original "
                f"maxLength for field {field_name!r}"
            )
        if done:
            return accepted


def _recover_source_edit_arguments(
    current: Any,
    adapter: Any,
    request: Any,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    from .model_adapters import ModelConfigurationError

    try:
        selector_schema = _source_edit_selector_schema(parameters)
    except ValueError as exc:
        raise ModelConfigurationError(str(exc)) from exc
    selector = _recover_page(
        current,
        adapter,
        request,
        page_index=1,
        page_count=2,
        page_schema=selector_schema,
        parameters=parameters,
        action_name=_SOURCE_EDIT_TOOL,
    )
    operation = _canonical_source_edit_operation(selector.get("operation"))
    try:
        detail_schema = _source_edit_detail_schema(parameters, operation)
    except ValueError as exc:
        raise ModelConfigurationError(str(exc)) from exc

    details: dict[str, Any] = {}
    scalar_schema = _source_edit_scalar_schema(detail_schema)
    if scalar_schema is not None:
        details.update(
            _recover_page(
                current,
                adapter,
                request,
                page_index=2,
                page_count=2,
                page_schema=scalar_schema,
                parameters=parameters,
                action_name=_SOURCE_EDIT_TOOL,
            )
        )

    detail_properties = detail_schema.get("properties")
    if not isinstance(detail_properties, Mapping):
        raise ModelConfigurationError(
            "HOST_ARGUMENT_DECOMPOSITION: apply_source_edit detail schema must expose properties"
        )
    required = set(_required_names(detail_schema))
    for field_name, raw_schema in detail_properties.items():
        if not isinstance(raw_schema, Mapping) or not _requires_stream_transport(raw_schema):
            continue
        if field_name not in required:
            continue
        details[str(field_name)] = _recover_source_edit_stream_field(
            current,
            adapter,
            request,
            operation=operation,
            field_name=str(field_name),
            field_schema=raw_schema,
        )

    merged = {"operation": operation, **details}
    forced = _forced_module()
    if not forced._arguments_match_schema(merged, parameters):
        raise ModelConfigurationError(
            "HOST_ARGUMENT_DECOMPOSITION: discriminated apply_source_edit arguments "
            "failed the original schema"
        )
    return merged


def host_selected_argument_turn(
    current: Any,
    adapter: Any,
    request: Any,
    name: str,
    *,
    prefix: str = "host_action",
) -> Any:
    forced = _forced_module()
    schema = forced._selected_schema(request, name)
    parameters = forced._parameters(schema)
    if name == _SOURCE_EDIT_TOOL:
        merged = _recover_source_edit_arguments(current, adapter, request, parameters)
        return forced._response_for_call(name, merged, prefix=prefix)

    try:
        pages = _argument_pages(parameters)
    except ValueError as exc:
        from .model_adapters import ModelConfigurationError

        raise ModelConfigurationError(str(exc)) from exc
    merged: dict[str, Any] = {}
    for page_index, page_schema in enumerate(pages, start=1):
        arguments = _recover_page(
            current,
            adapter,
            request,
            page_index=page_index,
            page_count=len(pages),
            page_schema=page_schema,
            parameters=parameters,
            action_name=name,
        )
        overlap = set(merged).intersection(arguments)
        if overlap:
            from .model_adapters import ModelConfigurationError

            raise ModelConfigurationError(
                "HOST_ARGUMENT_DECOMPOSITION: duplicate fields across pages: "
                + ", ".join(sorted(overlap))
            )
        merged.update(arguments)

    if not forced._arguments_match_schema(merged, parameters):
        from .model_adapters import ModelConfigurationError

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
