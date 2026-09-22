from __future__ import annotations

"""Recover verifier-repair candidates rejected before host atomic binding."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .source_repair_semantics import atomic_repair_scope_error
from .verifier_repair_window import (
    normalize_model_repair_replacement,
    repair_replacement_max_chars,
)

_MODEL_REJECTION_TOOL_NAME = "__mmm_rejected_tool_call__"


def model_tool_rejection_feedback(
    calls: Sequence[Any],
) -> tuple[str, list[Mapping[str, Any]]] | None:
    rejections: list[Mapping[str, Any]] = []
    for call in calls:
        if str(getattr(call, "name", "") or "").strip() != _MODEL_REJECTION_TOOL_NAME:
            continue
        arguments = getattr(call, "arguments", None)
        if isinstance(arguments, Mapping):
            rejections.append(dict(arguments))
        else:
            rejections.append({
                "failure_code": "MODEL_TOOL_CALL_REJECTED",
                "error": "invalid rejection payload",
            })
    if not rejections:
        return None

    details: list[str] = []
    for payload in rejections:
        code = str(payload.get("failure_code") or "MODEL_TOOL_CALL_REJECTED").strip()
        name = str(
            payload.get("original_tool")
            or payload.get("rejected_name")
            or ""
        ).strip()
        error = str(payload.get("error") or "").strip()
        line = code + (f" for {name!r}" if name else "")
        if error:
            line += f": {error}"
        details.append(line)
    return (
        "The previous model tool call was rejected during host admission and was not executed. "
        "Correct the tool name/arguments to match the currently exposed schema and try the required "
        "phase action again. Rejections: " + " | ".join(details),
        rejections,
    )


def recover_schema_rejected_host_bound_existing_calls(
    calls: Sequence[Any],
    *,
    phase: str,
    validation_status: str,
    context: Any,
) -> tuple[Any, ...] | None:
    """Strip redundant host-owned fields from one rejected existing-source call.

    Existing authored targets expose only the model-owned new source. Small models
    may still echo path/operation/old/count from prior tool turns. Those fields are safe
    to discard only when they agree with the already-pinned host target.
    """

    if (
        phase != "ACT"
        or validation_status == "FAIL"
        or len(calls) != 1
        or context is None
        or getattr(context, "is_new_file", False)
        or str(getattr(context, "evidence_source", "") or "") != "workspace_existing_target"
    ):
        return None
    target = str(getattr(context, "target_path", "") or "").replace("\\", "/").strip()
    if not target:
        return None
    call = calls[0]
    if str(getattr(call, "name", "") or "").strip() != _MODEL_REJECTION_TOOL_NAME:
        return None
    payload = getattr(call, "arguments", None)
    if not isinstance(payload, Mapping):
        return None
    if (
        str(payload.get("failure_code") or "") != "TOOL_SCHEMA_INVALID"
        or str(payload.get("original_tool") or "") != "apply_source_edit"
    ):
        return None
    raw_arguments = payload.get("raw_arguments")
    if not isinstance(raw_arguments, str) or not raw_arguments:
        return None
    try:
        candidate = json.loads(raw_arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(candidate, Mapping):
        return None
    allowed = {"new", "path", "operation", "old", "count"}
    if set(candidate) - allowed:
        return None
    model_new = candidate.get("new")
    if not isinstance(model_new, str) or not model_new:
        return None
    current_source = getattr(context, "source_body", None)
    model_old = candidate.get("old")
    if isinstance(model_old, str) and model_old:
        if not isinstance(current_source, str) or current_source.count(model_old) != 1:
            return None
        # Legacy/small adapters often keep emitting an exact old/new span after the
        # host has projected the schema down to {"new"}. Preserve that semantic
        # intent deterministically: apply the span to the host-owned live source,
        # then pass the resulting complete source through the normal new-only binder.
        if model_old != current_source:
            model_new = current_source.replace(model_old, model_new, 1)
    supplied_count = candidate.get("count")
    if supplied_count not in (None, 1, "1"):
        return None
    supplied_path = str(candidate.get("path") or "").replace("\\", "/").strip()
    while supplied_path.startswith("./"):
        supplied_path = supplied_path[2:]
    if supplied_path and supplied_path != target:
        return None
    operation = str(candidate.get("operation") or "").strip().casefold()
    if operation and operation not in {"replace", "replace_exact"}:
        return None

    arguments = {"new": model_new}
    return (
        replace(
            call,
            name="apply_source_edit",
            arguments=arguments,
            raw_arguments=json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )


def recover_schema_rejected_verifier_repair_calls(
    calls: Sequence[Any],
    *,
    phase: str,
    validation_status: str,
    context: Any,
    repair_window: Mapping[str, Any] | None,
) -> tuple[Any, ...] | None:
    """Salvage only provably atomic deltas from a schema-rejected repair call."""

    if (
        phase != "ACT"
        or validation_status != "FAIL"
        or len(calls) != 1
        or context is None
        or getattr(context, "is_new_file", False)
        or not isinstance(getattr(context, "source_body", None), str)
        or not repair_window
    ):
        return None
    call = calls[0]
    if str(getattr(call, "name", "") or "") != _MODEL_REJECTION_TOOL_NAME:
        return None
    payload = getattr(call, "arguments", None)
    if not isinstance(payload, Mapping):
        return None
    if (
        str(payload.get("failure_code") or "") != "TOOL_SCHEMA_INVALID"
        or str(payload.get("original_tool") or "") != "apply_source_edit"
    ):
        return None
    raw_arguments = payload.get("raw_arguments")
    if not isinstance(raw_arguments, str) or not raw_arguments:
        return None
    try:
        candidate = json.loads(raw_arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(candidate, Mapping) or set(candidate) != {"new"}:
        return None

    old_text = repair_window.get("old")
    model_new = candidate.get("new")
    current_source = getattr(context, "source_body", None)
    if not isinstance(old_text, str) or not old_text or not isinstance(model_new, str):
        return None
    local_new = normalize_model_repair_replacement(current_source, old_text, model_new)
    if local_new == model_new:
        return None
    if atomic_repair_scope_error(
        old_text=old_text,
        new_text=local_new,
        max_chars=repair_replacement_max_chars(old_text),
    ) is not None:
        return None

    arguments = {"new": local_new}
    return (
        replace(
            call,
            name="apply_source_edit",
            arguments=arguments,
            raw_arguments=json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        ),
    )
