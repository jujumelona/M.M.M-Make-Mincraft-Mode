from __future__ import annotations

"""Canonical model-facing source edit protocol.

One model action describes one executable edit. Existing files are changed through
exact spans or anchors; a genuinely new file may be created in one action. The host
resolves the project, validates the current file state, derives a SHA precondition and
compiles the action into the internal transactional ``apply_source_patch`` primitive.
"""

import hashlib
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_CANONICAL_OPERATIONS = (
    "replace_exact",
    "insert_before",
    "insert_after",
    "create_file",
    "delete_file",
)
_OPERATION_ALIASES = {
    "replace": "replace_exact",
    "create": "create_file",
    "delete": "delete_file",
}
_ACCEPTED_OPERATIONS = (*_CANONICAL_OPERATIONS, *_OPERATION_ALIASES)

SOURCE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation", "path"],
    "properties": {
        "operation": {
            "type": "string",
            "enum": list(_ACCEPTED_OPERATIONS),
            "description": (
                "Use replace_exact or insert_before/insert_after for an existing file, "
                "create_file for a new file, and delete_file only when removal is intended."
            ),
        },
        "path": {
            "type": "string",
            "minLength": 1,
            "description": "Project-relative source/resource path selected for this edit.",
        },
        "old": {
            "type": "string",
            "minLength": 1,
            "description": "Exact current span to replace for replace_exact.",
        },
        "new": {
            "type": "string",
            "description": "Replacement text for replace_exact.",
        },
        "anchor": {
            "type": "string",
            "minLength": 1,
            "description": "Exact current anchor for insert_before or insert_after.",
        },
        "content": {
            "type": "string",
            "description": "Inserted text, or complete content when creating a new file.",
        },
        "text": {
            "type": "string",
            "description": "Lossless Qwen alias for new/content.",
        },
        "count": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "Expected exact occurrence count for the selected span/anchor.",
        },
    },
}

_ALLOWED_FIELDS = frozenset(SOURCE_EDIT_SCHEMA["properties"])


def _required_text(
    runtime_module: Any,
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        requirement = "text" if allow_empty else "a non-empty string"
        raise runtime_module.AgentToolRuntimeError(f"{key} must be {requirement}")
    return value


def _normalize_operation(runtime_module: Any, value: Any) -> str:
    if not isinstance(value, str):
        raise runtime_module.AgentToolRuntimeError("operation must be a string")
    operation = _OPERATION_ALIASES.get(value.strip(), value.strip())
    if operation not in _CANONICAL_OPERATIONS:
        raise runtime_module.AgentToolRuntimeError(
            f"Unsupported source write operation: {value!r}"
        )
    return operation


def _canonicalize_text_alias(payload: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    if operation == "delete_file" or "text" not in payload:
        return payload
    canonical_key = "new" if operation == "replace_exact" else "content"
    if canonical_key in payload:
        return payload
    normalized = dict(payload)
    normalized[canonical_key] = payload["text"]
    return normalized


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
            f"Source edit requires an existing regular file: {normalized}"
        )
    return normalized, target


def _bound_project(
    runtime_module: Any,
    workspace_root: str | Path,
    bound_project_root: str | Path | None,
) -> tuple[Path, str]:
    if bound_project_root is None:
        return runtime_module._discover_model_project_root(workspace_root)

    workspace = Path(workspace_root).expanduser().resolve()
    root = Path(bound_project_root).expanduser().resolve()
    if (
        not workspace.is_dir()
        or workspace.is_symlink()
        or not root.is_dir()
        or root.is_symlink()
    ):
        raise runtime_module.AgentToolRuntimeError(
            "Host-bound model workspace/project must be regular directories"
        )
    try:
        relative = root.relative_to(workspace)
    except ValueError as exc:
        raise runtime_module.AgentToolRuntimeError(
            "Host-bound model project escaped its workspace"
        ) from exc
    return root, "." if not relative.parts else relative.as_posix()


def _replacement_for_edit(
    runtime_module: Any,
    *,
    operation: str,
    path: str,
    payload: Mapping[str, Any],
    count: int,
) -> dict[str, Any]:
    if operation == "replace_exact":
        old = _required_text(runtime_module, payload, "old")
        new = _required_text(runtime_module, payload, "new", allow_empty=True)
        return {"old": old, "new": new, "count": count}
    anchor = _required_text(runtime_module, payload, "anchor")
    content = _required_text(runtime_module, payload, "content", allow_empty=True)
    new = content + anchor if operation == "insert_before" else anchor + content
    return {"old": anchor, "new": new, "count": count}


def materialize_model_source_edit(
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
    *,
    bound_project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compile one semantic edit into one SHA-bound transactional host patch."""

    if not isinstance(payload, Mapping):
        raise runtime_module.AgentToolRuntimeError("Source-write arguments must be an object")
    extra = set(payload) - _ALLOWED_FIELDS
    if extra:
        raise runtime_module.AgentToolRuntimeError(
            f"Unknown model-facing source-write fields: {sorted(extra)}"
        )

    operation = _normalize_operation(runtime_module, payload.get("operation"))
    payload = _canonicalize_text_alias(payload, operation)
    path = _required_text(runtime_module, payload, "path")
    root, project_root_argument = _bound_project(
        runtime_module,
        workspace_root,
        bound_project_root,
    )
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
        content = _required_text(runtime_module, payload, "content", allow_empty=True)
        return {
            "project_root": project_root_argument,
            "operations": [
                {"operation": "create", "path": normalized, "content": content}
            ],
        }

    raw_bytes = target.read_bytes()
    expected_sha256 = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()

    if operation == "delete_file":
        forbidden = {"old", "new", "anchor", "content", "text", "count"}.intersection(payload)
        if forbidden:
            raise runtime_module.AgentToolRuntimeError(
                f"Fields {sorted(forbidden)} are invalid for delete_file"
            )
        return {
            "project_root": project_root_argument,
            "operations": [
                {
                    "operation": "delete",
                    "path": normalized,
                    "expected_sha256": expected_sha256,
                }
            ],
        }

    count = payload.get("count", 1)
    if type(count) is not int or count < 1:
        raise runtime_module.AgentToolRuntimeError("count must be a positive integer")

    if operation == "replace_exact":
        forbidden = {"anchor", "content"}.intersection(payload)
    else:
        forbidden = {"old", "new"}.intersection(payload)
    if forbidden:
        raise runtime_module.AgentToolRuntimeError(
            f"Fields {sorted(forbidden)} are invalid for {operation}"
        )

    replacement = _replacement_for_edit(
        runtime_module,
        operation=operation,
        path=normalized,
        payload=payload,
        count=count,
    )
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise runtime_module.AgentToolRuntimeError(
            f"Source edit target is not UTF-8 text: {normalized}"
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
                "expected_sha256": expected_sha256,
                "replacements": [replacement],
            }
        ],
    }


__all__ = ["SOURCE_EDIT_SCHEMA", "materialize_model_source_edit"]
