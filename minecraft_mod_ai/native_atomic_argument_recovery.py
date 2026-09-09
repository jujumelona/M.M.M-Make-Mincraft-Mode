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
_SOURCE_EDIT_TOOL = "apply_source_edit"
_SOURCE_EDIT_OPERATION_ALIASES = {
    "create": "create_file",
    "replace": "replace_exact",
    "delete": "delete_file",
}
_SOURCE_EDIT_OPERATION_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "replace_exact": (("path", "old", "new", "count"), ("path", "old", "new")),
    "insert_before": (("path", "anchor", "content", "count"), ("path", "anchor", "content")),
    "insert_after": (("path", "anchor", "content", "count"), ("path", "anchor", "content")),
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
            name: dict(properties[name]) if isinstance(properties[name], Mapping) else properties[name]
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


def _page_owned_arguments(
    arguments: Mapping[str, Any],
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep this page's fields while preserving truly unknown fields for rejection.

    A page is only a transport slice of the original function schema. The model still sees
    the original planning context and can therefore emit a property that is legal in the
    complete schema but owned by a later page. Such a field must not make the current page
    fail `additionalProperties: false`; its owning page remains responsible for producing
    and validating it. Fields absent from the complete schema are deliberately retained so
    the strict page validator still rejects genuine out-of-contract output.
    """

    page_properties = page_schema.get("properties")
    all_properties = parameters.get("properties")
    if not isinstance(page_properties, Mapping) or not isinstance(all_properties, Mapping):
        return dict(arguments)
    return {
        str(name): value
        for name, value in arguments.items()
        if name in page_properties or name not in all_properties
    }


def _page_result(
    turn: Any,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
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
    normalized = _page_owned_arguments(dict(parsed), page_schema, parameters)
    if not forced._arguments_match_schema(normalized, page_schema):
        diag = getattr(forced, "_schema_validation_diagnostics", lambda *args: "")(
            normalized, page_schema
        )
        reason = (
            f"argument page JSON failed the host page schema ({diag})"
            if diag
            else "argument page JSON failed the host page schema"
        )
        return None, reason, _fingerprint(normalized)
    return normalized, "", _fingerprint(normalized)


def _page_attempt(
    current: Any,
    adapter: Any,
    request: Any,
    page_schema: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str, str]:
    try:
        turn = current(adapter, request)
    except Exception as exc:
        from .llama_finish_reason_contract import completion_boundary_error
        from .generation_output_budget import GenerationOutputBudgetError

        # Backend/context failures belong to the canonical recovery owner. Retrying
        # them as invalid JSON loses their type, cause and preserved partial receipt.
        if completion_boundary_error(exc) is not None or isinstance(exc, GenerationOutputBudgetError):
            raise
        cause = getattr(exc, "cause", exc)
        reason = f"{type(cause).__name__}: {cause}"[:_MAX_REPAIR_ERROR_CHARS]
        return None, reason, _fingerprint({"exception": reason})
    return _page_result(turn, page_schema, parameters)


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
) -> dict[str, Any]:
    from .model_adapters import ModelConfigurationError

    first_request = _request(
        request,
        page_index=page_index,
        page_count=page_count,
        page_schema=page_schema,
    )
    arguments, error, first_fingerprint = _page_attempt(
        current,
        adapter,
        first_request,
        page_schema,
        parameters,
    )
    if arguments is not None:
        return arguments

    repair_request = _request(
        request,
        page_index=page_index,
        page_count=page_count,
        page_schema=page_schema,
        repair_error=error,
    )
    arguments, repair_error, second_fingerprint = _page_attempt(
        current,
        adapter,
        repair_request,
        page_schema,
        parameters,
    )
    if arguments is not None:
        return arguments

    fixed_point = first_fingerprint == second_fingerprint
    suffix = (
        "repeated-invalid argument-page fixed point"
        if fixed_point
        else "bounded argument-page repair exhausted"
    )
    raise ModelConfigurationError(
        f"Host-selected action {action_name!r} {suffix} on page "
        f"{page_index}/{page_count}; error={repair_error or error}."
    )


def _recover_source_edit_arguments(
    current: Any,
    adapter: Any,
    request: Any,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover the discriminator first, then only fields owned by that operation.

    SOURCE_EDIT_SCHEMA is deliberately a flat model-facing union. Treating that union as
    independent transport pages lets irrelevant optional fields from other operations become
    valid page output and leak into the merged call. The runtime semantic validator then has
    to reject an object the host itself assembled. Resolve the operation first and narrow the
    second page to its canonical semantic contract instead.
    """

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
    details = _recover_page(
        current,
        adapter,
        request,
        page_index=2,
        page_count=2,
        page_schema=detail_schema,
        parameters=parameters,
        action_name=_SOURCE_EDIT_TOOL,
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
    """Recover one already-selected action through bounded argument-only JSON pages."""

    from .model_adapters import ModelConfigurationError

    forced = _forced_module()
    parameters = forced._parameters(forced._selected_schema(request, name))

    if name == _SOURCE_EDIT_TOOL:
        merged = _recover_source_edit_arguments(current, adapter, request, parameters)
        return forced._response_for_call(name, merged, prefix=prefix)

    try:
        pages = _argument_pages(parameters)
    except ValueError as exc:
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
