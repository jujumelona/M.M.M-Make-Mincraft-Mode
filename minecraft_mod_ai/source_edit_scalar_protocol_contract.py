from __future__ import annotations

"""Bound the model-facing source-edit ACI to one scalar edit per tool turn.

Qwen tagged tool transports are much more reliable with scalar parameters than with a
large array of nested edit objects. The host still resolves the project, verifies the
current file hash and compiles the request into the canonical transactional
``apply_source_patch`` operation.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

_MAX_PATH_CHARS = 512
_MAX_MATCH_CHARS = 4096
_MAX_REPLACEMENT_CHARS = 8192
_MAX_PAYLOAD_BYTES = 16 * 1024
_MAX_COUNT = 16
_MARKER = "_mmm_scalar_source_edit_protocol_v1"
_CANONICAL_OPERATIONS = ("replace_exact", "insert_before", "insert_after")
_OPERATION_ALIASES = {"replace": "replace_exact"}
_ACCEPTED_OPERATIONS = (*_CANONICAL_OPERATIONS, *_OPERATION_ALIASES)

SOURCE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation", "path"],
    "properties": {
        "operation": {
            "type": "string",
            # Keep the model contract narrow while accepting the one common Qwen
            # shorthand that is losslessly canonicalized by the host before dispatch.
            "enum": list(_ACCEPTED_OPERATIONS),
        },
        "path": {"type": "string", "minLength": 1, "maxLength": _MAX_PATH_CHARS},
        "old": {"type": "string", "minLength": 1, "maxLength": _MAX_MATCH_CHARS},
        "new": {"type": "string", "maxLength": _MAX_REPLACEMENT_CHARS},
        "anchor": {"type": "string", "minLength": 1, "maxLength": _MAX_MATCH_CHARS},
        "content": {"type": "string", "maxLength": _MAX_REPLACEMENT_CHARS},
        "count": {
            "type": "integer",
            "minimum": 1,
            "maximum": _MAX_COUNT,
            "default": 1,
        },
    },
}

_ALLOWED_FIELDS = frozenset(SOURCE_EDIT_SCHEMA["properties"])


def _bounded_text(
    runtime_module: Any,
    payload: Mapping[str, Any],
    key: str,
    *,
    maximum: int,
    required: bool = False,
) -> str | None:
    value = payload.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value):
        requirement = "a non-empty string" if required else "text"
        raise runtime_module.AgentToolRuntimeError(f"{key} must be {requirement}")
    if len(value) > maximum:
        raise runtime_module.AgentToolRuntimeError(
            f"{key} exceeds the model-facing {maximum}-character limit"
        )
    return value


def _normalize_operation(runtime_module: Any, value: Any) -> str:
    if not isinstance(value, str):
        raise runtime_module.AgentToolRuntimeError("operation must be a string")
    operation = value.strip()
    operation = _OPERATION_ALIASES.get(operation, operation)
    if operation not in _CANONICAL_OPERATIONS:
        raise runtime_module.AgentToolRuntimeError(
            f"Unsupported source edit operation: {value!r}"
        )
    return operation


def materialize_model_source_edit(
    extension_module: Any,
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile exactly one scalar edit into one SHA-guarded canonical patch."""

    if not isinstance(payload, Mapping):
        raise runtime_module.AgentToolRuntimeError("Source-edit arguments must be an object")
    extra = set(payload) - _ALLOWED_FIELDS
    if extra:
        raise runtime_module.AgentToolRuntimeError(
            f"Unknown model-facing source-edit fields: {sorted(extra)}"
        )
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    if len(encoded) > _MAX_PAYLOAD_BYTES:
        raise runtime_module.AgentToolRuntimeError(
            f"Source-edit arguments exceed the {_MAX_PAYLOAD_BYTES}-byte turn limit"
        )

    operation = _normalize_operation(runtime_module, payload.get("operation"))
    path = _bounded_text(
        runtime_module, payload, "path", maximum=_MAX_PATH_CHARS, required=True
    )
    count = payload.get("count", 1)
    if type(count) is not int or not 1 <= count <= _MAX_COUNT:
        raise runtime_module.AgentToolRuntimeError(
            f"count must be an integer between 1 and {_MAX_COUNT}"
        )

    item: dict[str, Any] = {"operation": operation, "path": path, "count": count}
    if operation == "replace_exact":
        item["old"] = _bounded_text(
            runtime_module,
            payload,
            "old",
            maximum=_MAX_MATCH_CHARS,
            required=True,
        )
        item["new"] = _bounded_text(
            runtime_module,
            payload,
            "new",
            maximum=_MAX_REPLACEMENT_CHARS,
            required=True,
        )
        forbidden = {"anchor", "content"}.intersection(payload)
    else:
        item["anchor"] = _bounded_text(
            runtime_module,
            payload,
            "anchor",
            maximum=_MAX_MATCH_CHARS,
            required=True,
        )
        item["content"] = _bounded_text(
            runtime_module,
            payload,
            "content",
            maximum=_MAX_REPLACEMENT_CHARS,
            required=True,
        )
        forbidden = {"old", "new"}.intersection(payload)
    if forbidden:
        raise runtime_module.AgentToolRuntimeError(
            f"Fields {sorted(forbidden)} are invalid for {operation}"
        )

    root, project_root_argument = runtime_module._discover_model_project_root(workspace_root)
    normalized, target = extension_module._normalize_model_source_path(
        runtime_module, root, path
    )
    replacement = extension_module._replacement_for_edit(
        runtime_module, item, normalized
    )
    raw_bytes = target.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise runtime_module.AgentToolRuntimeError(
            f"Partial source edit target is not UTF-8 text: {normalized}"
        ) from exc
    found = text.count(replacement["old"])
    if found != replacement["count"]:
        raise runtime_module.AgentToolRuntimeError(
            f"Exact source-edit precondition failed for {normalized}: expected "
            f"{replacement['count']} matches, found {found}"
        )

    return {
        "project_root": project_root_argument,
        "operations": [
            {
                "operation": "edit",
                "path": normalized,
                "expected_sha256": "sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
                "replacements": [replacement],
            }
        ],
    }


def install(extension_module: Any, runtime_module: Any) -> None:
    if bool(getattr(extension_module, _MARKER, False)):
        return

    def materialize(
        runtime_owner: Any,
        workspace_root: str | Path,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        return materialize_model_source_edit(
            extension_module,
            runtime_owner,
            workspace_root,
            payload,
        )

    extension_module._SOURCE_EDIT_SCHEMA = SOURCE_EDIT_SCHEMA
    extension_module._materialize_model_source_edit = materialize
    setattr(extension_module, _MARKER, True)

    # Existing AgentToolRuntime instances may have cached the old nested-array schema.
    # Finalization normally installs before instances exist, but clearing class-owned
    # caches here would be unsafe because caches are per instance. The runtime schema
    # wrapper reads this module global on its first generation-schema construction.
    del runtime_module


__all__ = ["SOURCE_EDIT_SCHEMA", "install", "materialize_model_source_edit"]
