from __future__ import annotations

"""Harden existing-file repair turns and JDT verifier-readiness semantics.

The normal tool schema intentionally supports both creation and editing because a coder
can legitimately begin a fresh task. Once VERIFY has produced trustworthy source
diagnostics for an existing target, however, the next ACT turn is a repair turn. At
that point creation/deletion operations and target drift are not valid choices and are
removed from the model-visible causal frontier while the broader host validation
surface remains unchanged.

JDT core-runtime/classpath failures are verifier-health failures, not source defects.
They therefore retire the unhealthy verifier for the current HostRunState instead of
feeding bogus diagnostics back into source repair.
"""

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from functools import wraps
from typing import Any

_MARKER = "_mmm_repair_mutation_recovery_v1"
_REPAIR_PREFIX = "MMM_CORE_VERIFIER_REPAIR_V1"
_REPAIR_FORBIDDEN_OPERATIONS = frozenset(
    {
        "create",
        "create_file",
        "create_java_type",
        "delete",
        "delete_file",
    }
)
_JDT_TOOLS = frozenset({"java_diagnostics", "jdt_diagnostics"})


def _tool_name(schema: Any) -> str:
    if not isinstance(schema, Mapping):
        return ""
    function = schema.get("function")
    if not isinstance(function, Mapping):
        return ""
    return str(function.get("name") or "").strip()


def _existing_repair_context(
    messages: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return the latest host repair receipt only for an existing pinned target."""

    for message in reversed(messages):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").strip().casefold() != "system":
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(_REPAIR_PREFIX):
            continue
        raw = content.rsplit("\n", 1)[-1].strip()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(payload, Mapping):
            return None
        if payload.get("target_is_new_file") is not False:
            return None
        path = str(payload.get("target_path") or "").strip().replace("\\", "/")
        if not path:
            return None
        return payload
    return None


def _constrain_existing_repair_schema(
    schema: Mapping[str, Any], *, target_path: str
) -> Mapping[str, Any]:
    """Clone one model-visible source-edit schema and remove illegal repair choices."""

    cloned = copy.deepcopy(dict(schema))
    function = cloned.get("function")
    if not isinstance(function, dict):
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no function schema"
        )
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no parameter schema"
        )
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no properties schema"
        )

    operation = properties.get("operation")
    if not isinstance(operation, dict):
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no operation schema"
        )
    values = operation.get("enum")
    if not isinstance(values, list):
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit operation is not enumerated"
        )
    allowed = [
        value
        for value in values
        if str(value).strip().casefold() not in _REPAIR_FORBIDDEN_OPERATIONS
    ]
    if not allowed:
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: no existing-file repair operations remain"
        )
    operation["enum"] = allowed
    operation["description"] = (
        str(operation.get("description") or "").rstrip()
        + " Existing-file repair turn: creation and deletion operations are structurally unavailable."
    ).strip()

    path_schema = properties.get("path")
    if not isinstance(path_schema, dict):
        raise RuntimeError(
            "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit has no path schema"
        )
    path_schema["enum"] = [target_path]
    path_schema["description"] = (
        str(path_schema.get("description") or "").rstrip()
        + " This repair turn is pinned to the existing host target."
    ).strip()
    return cloned


def constrain_existing_repair_tools(
    tools: Sequence[Any],
    messages: Sequence[Mapping[str, Any]],
) -> tuple[Any, ...]:
    """Narrow only the model-visible repair frontier; never mutate canonical schemas."""

    context = _existing_repair_context(messages)
    if context is None:
        return tuple(tools)
    target_path = str(context.get("target_path") or "").strip().replace("\\", "/")
    result: list[Any] = []
    found = False
    for schema in tools:
        if _tool_name(schema) == "apply_source_edit":
            if not isinstance(schema, Mapping):
                raise RuntimeError(
                    "REPAIR_TOOL_SCHEMA_INVALID: apply_source_edit schema is not a mapping"
                )
            result.append(
                _constrain_existing_repair_schema(schema, target_path=target_path)
            )
            found = True
        else:
            result.append(schema)
    return tuple(result) if found else tuple(tools)


def jdt_workspace_not_ready(payload: Mapping[str, Any]) -> bool:
    """Return whether a successful JDT transport proves an unready Java workspace."""

    result = payload.get("result")
    if not isinstance(result, Mapping):
        return False
    from .validation_diagnostic_contract import diagnostic_errors

    return any(
        isinstance(item, Mapping)
        and str(item.get("code") or "").strip().upper() == "JDT_WORKSPACE_NOT_READY"
        for item in diagnostic_errors(result)
    )


def install(loop_module: Any | None = None) -> None:
    """Install before the final semantic-failure guard so that guard remains outermost."""

    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if getattr(loop_module, _MARKER, False):
        return

    original_generate = getattr(loop_module, "_generate_turn_with_context_recovery", None)
    original_outcome = getattr(loop_module, "_verification_outcome", None)
    if not callable(original_generate) or not callable(original_outcome):
        raise RuntimeError(
            "repair recovery requires the progress-aware generation and verification boundaries"
        )

    @wraps(original_generate)
    def generate_with_repair_constraints(*args: Any, **kwargs: Any):
        request = kwargs.get("request")
        messages = kwargs.get("messages")
        if (
            request is not None
            and isinstance(messages, Sequence)
            and not isinstance(messages, (str, bytes, bytearray))
        ):
            constrained = constrain_existing_repair_tools(request.tools, messages)
            if constrained != tuple(request.tools):
                request = replace(request, tools=constrained)
                kwargs["request"] = request
                emit = getattr(loop_module, "emit_root_cause", None)
                if callable(emit):
                    emit(
                        "repair_tool_frontier_constrained",
                        stage="generation",
                        operation="apply_source_edit",
                        gate="repair_mutation_schema",
                        result="PASS",
                        reason="existing-file repair removed create/delete operations and pinned target path",
                        details={
                            "selected_tools": [
                                _tool_name(schema) for schema in constrained if _tool_name(schema)
                            ]
                        },
                    )
        return original_generate(*args, **kwargs)

    @wraps(original_outcome)
    def verification_outcome(tool_name: str, payload: Mapping[str, Any]) -> str:
        outcome = original_outcome(tool_name, payload)
        if (
            outcome == "FAIL"
            and str(tool_name or "").strip() in _JDT_TOOLS
            and jdt_workspace_not_ready(payload)
        ):
            emit = getattr(loop_module, "emit_root_cause", None)
            if callable(emit):
                emit(
                    "jdt_workspace_readiness_reclassified",
                    stage="generation",
                    operation=str(tool_name),
                    gate="verifier_semantics",
                    result="UNAVAILABLE",
                    reason="Java core runtime symbols are unresolved; source repair suppressed",
                )
            return "UNAVAILABLE"
        return outcome

    loop_module._generate_turn_with_context_recovery = generate_with_repair_constraints
    loop_module._verification_outcome = verification_outcome
    setattr(loop_module, _MARKER, True)


def assert_installed(loop_module: Any | None = None) -> None:
    if loop_module is None:
        from . import progress_aware_tool_loop as loop_module
    if not getattr(loop_module, _MARKER, False):
        raise RuntimeError("repair mutation recovery contract is not installed")


__all__ = [
    "assert_installed",
    "constrain_existing_repair_tools",
    "install",
    "jdt_workspace_not_ready",
]
