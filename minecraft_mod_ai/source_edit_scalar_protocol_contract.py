from __future__ import annotations

"""Bound the model-facing source-write ACI to one scalar action per tool turn.

Qwen tagged tool transports are much more reliable with scalar parameters than with a
large array of nested edit objects. The host still resolves the project, validates the
project-relative target, verifies exact-match preconditions for existing files and
compiles the request into the canonical transactional ``apply_source_patch`` operation.
New files use the same bounded scalar surface and are compiled to a host-owned create
operation rather than forcing the model onto the larger nested patch schema.
"""

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

_MAX_PATH_CHARS = 512
_MAX_MATCH_CHARS = 4096
_MAX_REPLACEMENT_CHARS = 8192
_MAX_PAYLOAD_BYTES = 16 * 1024
_MAX_COUNT = 16
_MARKER = "_mmm_scalar_source_edit_protocol_v1"
_CANONICAL_OPERATIONS = (
    "replace_exact",
    "insert_before",
    "insert_after",
    "create_file",
)
_OPERATION_ALIASES = {
    "replace": "replace_exact",
    "create": "create_file",
}
_ACCEPTED_OPERATIONS = (*_CANONICAL_OPERATIONS, *_OPERATION_ALIASES)

SOURCE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation", "path"],
    "properties": {
        "operation": {
            "type": "string",
            "description": (
                "Use create_file only when the target does not exist. Use replace_exact "
                "or insert_before/insert_after for an existing file."
            ),
            # Keep the model contract narrow while accepting common Qwen shorthands
            # that are losslessly canonicalized by the host before dispatch.
            "enum": list(_ACCEPTED_OPERATIONS),
        },
        "path": {"type": "string", "minLength": 1, "maxLength": _MAX_PATH_CHARS},
        "old": {"type": "string", "minLength": 1, "maxLength": _MAX_MATCH_CHARS},
        "new": {"type": "string", "maxLength": _MAX_REPLACEMENT_CHARS},
        "anchor": {"type": "string", "minLength": 1, "maxLength": _MAX_MATCH_CHARS},
        "content": {
            "type": "string",
            "maxLength": _MAX_REPLACEMENT_CHARS,
            "description": (
                "Inserted text for insert_before/insert_after, or complete bounded "
                "content for create_file."
            ),
        },
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


def _bounded_create_content(runtime_module: Any, payload: Mapping[str, Any]) -> str:
    value = payload.get("content")
    if not isinstance(value, str):
        raise runtime_module.AgentToolRuntimeError("create_file requires text content")
    if len(value) > _MAX_REPLACEMENT_CHARS:
        raise runtime_module.AgentToolRuntimeError(
            f"content exceeds the model-facing {_MAX_REPLACEMENT_CHARS}-character limit"
        )
    return value


def _normalize_operation(runtime_module: Any, value: Any) -> str:
    if not isinstance(value, str):
        raise runtime_module.AgentToolRuntimeError("operation must be a string")
    operation = value.strip()
    operation = _OPERATION_ALIASES.get(operation, operation)
    if operation not in _CANONICAL_OPERATIONS:
        raise runtime_module.AgentToolRuntimeError(
            f"Unsupported source write operation: {value!r}"
        )
    return operation


def _normalize_model_target(
    runtime_module: Any,
    root: Path,
    raw_path: Any,
    *,
    require_existing: bool,
) -> tuple[str, Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise runtime_module.AgentToolRuntimeError("Model source path must be a non-empty string")
    normalized = PurePosixPath(raw_path.strip().replace("\\", "/")).as_posix()
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized in {"", "."} or ".." in path.parts:
        raise runtime_module.AgentToolRuntimeError(f"Unsafe model source path: {raw_path!r}")
    if not any(normalized.startswith(prefix) for prefix in runtime_module._MODEL_SOURCE_PREFIXES):
        raise runtime_module.AgentToolRuntimeError(
            "Model source writes are limited to src/main/java, src/main/resources, "
            f"src/test/java and src/gametest: {normalized}"
        )

    cursor = root
    for part in path.parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise runtime_module.AgentToolRuntimeError(
                f"Model source path traverses a symlink: {normalized}"
            )
    target = root.joinpath(*path.parts)
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise runtime_module.AgentToolRuntimeError(
                f"Model source target must be a regular file: {normalized}"
            )
    elif require_existing:
        raise runtime_module.AgentToolRuntimeError(
            f"Partial source edit requires an existing regular file: {normalized}"
        )
    return normalized, target


def materialize_model_source_edit(
    extension_module: Any,
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile exactly one scalar source write into one canonical host patch."""

    if not isinstance(payload, Mapping):
        raise runtime_module.AgentToolRuntimeError("Source-write arguments must be an object")
    extra = set(payload) - _ALLOWED_FIELDS
    if extra:
        raise runtime_module.AgentToolRuntimeError(
            f"Unknown model-facing source-write fields: {sorted(extra)}"
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
            f"Source-write arguments exceed the {_MAX_PAYLOAD_BYTES}-byte turn limit"
        )

    operation = _normalize_operation(runtime_module, payload.get("operation"))
    path = _bounded_text(
        runtime_module, payload, "path", maximum=_MAX_PATH_CHARS, required=True
    )
    root, project_root_argument = runtime_module._discover_model_project_root(workspace_root)
    normalized, target = _normalize_model_target(
        runtime_module,
        root,
        path,
        require_existing=operation != "create_file",
    )

    if operation == "create_file":
        forbidden = {"old", "new", "anchor", "count"}.intersection(payload)
        if forbidden:
            raise runtime_module.AgentToolRuntimeError(
                f"Fields {sorted(forbidden)} are invalid for create_file"
            )
        if target.exists():
            raise runtime_module.AgentToolRuntimeError(
                f"create_file target already exists: {normalized}"
            )
        return {
            "project_root": project_root_argument,
            "operations": [
                {
                    "operation": "create",
                    "path": normalized,
                    "content": _bounded_create_content(runtime_module, payload),
                }
            ],
        }

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
