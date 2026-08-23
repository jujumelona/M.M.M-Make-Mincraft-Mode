from __future__ import annotations

"""Keep completion-boundary recovery on the canonical scalar source-edit protocol.

The normal ``apply_source_edit`` surface moved Java creation to structural actions
(``create_java_type``, ``add_java_import``, ``insert_java_member``).  Completion-boundary
recovery used to keep a private legacy action vocabulary with ``append_file`` and
whole-file Java ``create_file``.  That split made a truncated action recover into a
schema-valid value that the canonical materializer could never execute.

This late contract does not introduce another writer.  It narrows the recovery schema
and sanitized partial hints to the already-authoritative source-edit materializer, and
rewrites only continuation guidance so the model cannot be instructed to call retired
operations.
"""

import json
from typing import Any, Mapping

from .runtime_contract_wrappers import contract_wraps, has_contract_marker

_MARKER = "_mmm_canonical_bounded_scalar_recovery_v1"
_STRUCTURAL_JAVA = (
    "create_java_type",
    "add_java_import",
    "insert_java_member",
)
_BOUNDED_EXISTING = (
    "replace_exact",
    "insert_before",
    "insert_after",
)
_NON_JAVA_CREATE = ("create_file",)
_SAFE_OPERATIONS = (*_BOUNDED_EXISTING, *_NON_JAVA_CREATE, *_STRUCTURAL_JAVA)
_REQUIRED_BY_OPERATION = {
    "replace_exact": ("old", "new"),
    "insert_before": ("anchor", "content"),
    "insert_after": ("anchor", "content"),
    "create_file": ("content",),
    "create_java_type": ("package_name", "declaration"),
    "add_java_import": ("import_name",),
    "insert_java_member": ("member",),
}
_PAYLOAD_FIELDS = (
    "old",
    "new",
    "anchor",
    "content",
    "text",
    "package_name",
    "declaration",
    "import_name",
    "member",
)
_ALIASES = {
    "replace": "replace_exact",
    "create": "create_file",
}


def _operations_for_path(path: str) -> tuple[str, ...]:
    normalized = str(path or "").strip().casefold()
    if not normalized:
        return _SAFE_OPERATIONS
    if normalized.endswith(".java"):
        return (*_BOUNDED_EXISTING, *_STRUCTURAL_JAVA)
    return (*_BOUNDED_EXISTING, *_NON_JAVA_CREATE)


def _sanitize_hint(module: Any, hint: Mapping[str, Any]) -> dict[str, str]:
    path = str(hint.get("path") or "").strip()
    operation = _ALIASES.get(
        str(hint.get("operation") or "").strip(),
        str(hint.get("operation") or "").strip(),
    )
    result: dict[str, str] = {}
    if path:
        try:
            normalized = module._canonicalize_planned_path(path)
        except module.CustomModuleGenerationError:
            normalized = ""
        if (
            normalized
            and module._agent_mutable_path(normalized)
            and not module.custom_module_path_protected(normalized)
        ):
            result["path"] = normalized
    allowed = _operations_for_path(result.get("path", ""))
    if operation in allowed:
        result["operation"] = operation
    return result


def _partial_source_edit_hint(module: Any, boundary: Any) -> dict[str, str]:
    """Keep only completed executable headers from a truncated tool call.

    A legacy ``create_file``/``append_file`` header aimed at ``*.java`` contributes a
    safe path receipt but not an operation receipt.  Recovery can then select the
    structural Java action that the current canonical tool schema actually accepts.
    """

    partial = getattr(boundary, "partial_message", {})
    if not isinstance(partial, Mapping):
        return {}
    hints: list[dict[str, str]] = []
    for field in ("reasoning_content", "reasoning", "content"):
        text = partial.get(field)
        if not isinstance(text, str):
            continue
        match = module._PARTIAL_FUNCTION.search(text)
        if match is None:
            continue
        fragment = text[match.end() :]
        payload_starts = [
            position
            for key in _PAYLOAD_FIELDS
            if (position := fragment.find(f"<parameter={key}>")) >= 0
        ]
        header = fragment[: min(payload_starts)] if payload_starts else fragment
        raw_hint = {
            "operation": module._completed_partial_parameter(
                header, "operation", maximum=32
            ),
            "path": module._completed_partial_parameter(
                header, "path", maximum=512
            ),
        }
        hint = _sanitize_hint(module, raw_hint)
        if hint:
            hints.append(hint)
    if not hints:
        return {}
    result = dict(hints[0])
    for hint in hints[1:]:
        for key in tuple(result):
            if key in hint and hint[key] != result[key]:
                result.pop(key, None)
    return result


def _bounded_scalar_obligation_schema(module: Any, hint: Mapping[str, Any]) -> dict[str, Any]:
    """Return one bounded subset of the canonical scalar source-edit schema."""

    active_hint = _sanitize_hint(module, hint)
    path_value = active_hint.get("path", "")
    operations = _operations_for_path(path_value)
    operation: dict[str, Any] = {
        "type": "string",
        "enum": list(operations),
    }
    path: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 512,
    }
    hinted_operation = active_hint.get("operation", "")
    if hinted_operation in operations:
        operation["const"] = hinted_operation
    if path_value:
        path["const"] = path_value

    required = ["operation", "path"]
    if hinted_operation in _REQUIRED_BY_OPERATION:
        required.extend(_REQUIRED_BY_OPERATION[hinted_operation])

    chunk = int(module._RECOVERY_CHUNK_CHARS)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "operation": operation,
            "path": path,
            "old": {"type": "string", "minLength": 1, "maxLength": chunk},
            "new": {"type": "string", "maxLength": chunk},
            "anchor": {"type": "string", "minLength": 1, "maxLength": chunk},
            "content": {"type": "string", "minLength": 1, "maxLength": chunk},
            "count": {"type": "integer", "const": 1, "default": 1},
            "package_name": {"type": "string", "minLength": 1, "maxLength": 512},
            "declaration": {"type": "string", "minLength": 1, "maxLength": chunk},
            "import_name": {"type": "string", "minLength": 1, "maxLength": 512},
            "member": {"type": "string", "minLength": 1, "maxLength": chunk},
        },
    }
    if hinted_operation not in _REQUIRED_BY_OPERATION:
        schema["allOf"] = [
            {
                "if": {
                    "properties": {"operation": {"const": name}},
                    "required": ["operation"],
                },
                "then": {"required": list(_REQUIRED_BY_OPERATION[name])},
            }
            for name in operations
        ]
    return schema


def _canonical_continuation_rules(module: Any) -> list[str]:
    chunk = int(module._RECOVERY_CHUNK_CHARS)
    return [
        "Inspect the current workspace state and preserve every correct existing edit.",
        "The next tool call must be exactly one apply_source_edit action for exactly one project-relative path.",
        "Keep each apply_source_edit call to one bounded semantic action; never emit multiple files in one tool call.",
        "For a new Java file, use create_java_type for the empty type shell, then add_java_import and insert_java_member across later tool turns.",
        "Use create_file only for a complete new non-Java resource; Java whole-file create_file is invalid.",
        f"Keep bounded old/new/anchor/content/member values at or below {chunk} characters during recovery.",
        "For an existing file, prefer replace_exact, insert_before or insert_after over whole-file replacement.",
        "Do not repeat the truncated action. Do not put source code in the final summary.",
        "Continue tool turns until the approved module is implemented, then return only a concise summary.",
    ]


def install(module: Any) -> None:
    """Bind completion-boundary recovery to the canonical source-edit vocabulary."""

    current_hint = module._partial_source_edit_hint
    if not has_contract_marker(current_hint, _MARKER):

        @contract_wraps(current_hint)
        def partial_source_edit_hint(boundary: Any) -> dict[str, str]:
            return _partial_source_edit_hint(module, boundary)

        setattr(partial_source_edit_hint, _MARKER, True)
        module._partial_source_edit_hint = partial_source_edit_hint

    current_schema = module._bounded_scalar_obligation_schema
    if not has_contract_marker(current_schema, _MARKER):

        @contract_wraps(current_schema)
        def bounded_scalar_obligation_schema(hint: dict[str, str]) -> dict[str, Any]:
            return _bounded_scalar_obligation_schema(module, hint)

        setattr(bounded_scalar_obligation_schema, _MARKER, True)
        module._bounded_scalar_obligation_schema = bounded_scalar_obligation_schema

    current_continuation = module._output_exhaustion_continuation_messages
    if not has_contract_marker(current_continuation, _MARKER):

        @contract_wraps(current_continuation)
        def output_exhaustion_continuation_messages(*args: Any, **kwargs: Any):
            messages = current_continuation(*args, **kwargs)
            if not isinstance(messages, list):
                return messages
            updated = [dict(message) for message in messages]
            for message in reversed(updated):
                if message.get("role") != "user" or not isinstance(message.get("content"), str):
                    continue
                try:
                    payload = json.loads(message["content"])
                except json.JSONDecodeError:
                    break
                if not isinstance(payload, dict) or payload.get("phase") != "implement_module":
                    break
                payload["rules"] = _canonical_continuation_rules(module)
                message["content"] = json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                break
            return updated

        setattr(output_exhaustion_continuation_messages, _MARKER, True)
        module._output_exhaustion_continuation_messages = (
            output_exhaustion_continuation_messages
        )

    module._RECOVERY_SOURCE_EDIT_OPERATIONS = _SAFE_OPERATIONS


__all__ = ["install"]
