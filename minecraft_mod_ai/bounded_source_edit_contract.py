from __future__ import annotations

"""Bound the model-facing source-edit ACI while keeping host patching transactional.

Large repositories are external state. A coder turn should localize a target span,
submit one small anchored edit, observe the host receipt, and continue. Requiring the
model to re-emit complete files makes action size proportional to repository/file size
and can exhaust a local model's decode budget before a valid tool call closes.

The canonical first-party ``apply_source_patch`` protocol remains unchanged: the host
still owns project-root resolution, exact SHA-256 preconditions and transactional file
replacement. This contract changes only the model-facing action representation.
"""

import hashlib
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

_MAX_FILES = 4
_MAX_EDITS_PER_FILE = 8
_MAX_OLD_TEXT_BYTES = 4 * 1024
_MAX_NEW_TEXT_BYTES = 8 * 1024
_MAX_ACTION_TEXT_BYTES = 20 * 1024
_MARKER = "_mmm_bounded_source_edit_aci_v1"
_SCHEMA_MARKER = "_mmm_bounded_source_edit_schema_projection_v1"
_MODEL_DESCRIPTION = (
    "Apply a small exact-span source/resource edit. Retrieve or localize the current "
    "target first; existing files use unique old_text -> new_text replacements. Do not "
    "reproduce complete existing files. The host derives the bound project root, exact "
    "SHA-256 precondition and transactional patch. If more work remains, observe this "
    "receipt and continue with another bounded edit turn."
)

BOUNDED_SOURCE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["files"],
    "properties": {
        "files": {
            "type": "array",
            "minItems": 1,
            "maxItems": _MAX_FILES,
            "description": (
                "Small source edits only. Retrieve/localize the current target span first; "
                "do not reproduce complete existing files. If more work remains, make "
                "another edit action after observing this action's receipt."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "edits"],
                "properties": {
                    "path": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                            "Project-relative semantic source/resource path. Only src/main/java, "
                            "src/main/resources, src/test/java and src/gametest are writable."
                        ),
                    },
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": _MAX_EDITS_PER_FILE,
                        "description": (
                            "Ordered exact-span replacements. For an existing file, old_text "
                            "must be a non-empty span that occurs exactly once in the current "
                            "file. For a new file use exactly one edit with old_text empty."
                        ),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["old_text", "new_text"],
                            "properties": {
                                "old_text": {
                                    "type": "string",
                                    "description": (
                                        "Exact unique current span to replace. Keep it narrow; "
                                        "retrieve the relevant file/symbol again if unsure."
                                    ),
                                },
                                "new_text": {
                                    "type": "string",
                                    "description": "Replacement text for only that localized span.",
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def _byte_len(value: str) -> int:
    return len(value.encode("utf-8"))


def _validated_relative_path(runtime_module: Any, raw_path: Any) -> tuple[str, PurePosixPath]:
    error = runtime_module.AgentToolRuntimeError
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise error("Model source path must be a non-empty string")
    normalized = PurePosixPath(raw_path.strip().replace("\\", "/")).as_posix()
    path = PurePosixPath(normalized)
    if path.is_absolute() or normalized in {"", "."} or ".." in path.parts:
        raise error(f"Unsafe model source path: {raw_path!r}")
    if not any(normalized.startswith(prefix) for prefix in runtime_module._MODEL_SOURCE_PREFIXES):
        raise error(
            "Model source writes are limited to src/main/java, src/main/resources, "
            f"src/test/java and src/gametest: {normalized}"
        )
    return normalized, path


def _regular_target(runtime_module: Any, root: Path, path: PurePosixPath, normalized: str) -> Path:
    error = runtime_module.AgentToolRuntimeError
    cursor = root
    for part in path.parts[:-1]:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise error(f"Model source path traverses a symlink: {normalized}")
    target = root.joinpath(*path.parts)
    if target.exists() and (target.is_symlink() or not target.is_file()):
        raise error(f"Model source target must be a regular file: {normalized}")
    return target


def _edit_rows(runtime_module: Any, item: Mapping[str, Any]) -> list[tuple[str, str]]:
    error = runtime_module.AgentToolRuntimeError
    extra = set(item) - {"path", "edits"}
    if extra:
        raise error(
            "Model source files accept only path and edits; host-owned patch fields "
            f"are forbidden: {sorted(extra)}"
        )
    raw = item.get("edits")
    if not isinstance(raw, list) or not raw:
        raise error("Each model source file requires a non-empty edits list")
    if len(raw) > _MAX_EDITS_PER_FILE:
        raise error(f"edits exceeds the {_MAX_EDITS_PER_FILE}-edit per-file action limit")

    rows: list[tuple[str, str]] = []
    for edit in raw:
        if not isinstance(edit, Mapping):
            raise error("Each source edit must be an object")
        edit_extra = set(edit) - {"old_text", "new_text"}
        if edit_extra:
            raise error(f"Source edits accept only old_text and new_text: {sorted(edit_extra)}")
        old_text = edit.get("old_text")
        new_text = edit.get("new_text")
        if not isinstance(old_text, str) or not isinstance(new_text, str):
            raise error("old_text and new_text must be UTF-8 text")
        if _byte_len(old_text) > _MAX_OLD_TEXT_BYTES:
            raise error(
                f"old_text exceeds {_MAX_OLD_TEXT_BYTES} bytes; localize a narrower unique span"
            )
        if _byte_len(new_text) > _MAX_NEW_TEXT_BYTES:
            raise error(
                f"new_text exceeds {_MAX_NEW_TEXT_BYTES} bytes; split the change across turns"
            )
        rows.append((old_text, new_text))
    return rows


def materialize_bounded_source_edit(
    runtime_module: Any,
    workspace_root: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile bounded anchored edits into the host-owned transactional patch protocol."""

    error = runtime_module.AgentToolRuntimeError
    extra = set(payload) - {"files"}
    if extra:
        raise error(
            "Model-facing source writes accept only files; host-owned project/patch "
            f"fields are forbidden: {sorted(extra)}"
        )
    root, project_root_argument = runtime_module._discover_model_project_root(workspace_root)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise error("files must be a non-empty list")
    if len(raw_files) > _MAX_FILES:
        raise error(
            f"files exceeds the {_MAX_FILES}-file action limit; continue in another agent turn"
        )

    operations: list[dict[str, Any]] = []
    seen: set[str] = set()
    action_text_bytes = 0
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise error("Each model source write must be an object")
        normalized, path = _validated_relative_path(runtime_module, item.get("path"))
        if normalized in seen:
            raise error(f"Duplicate model source path: {normalized}")
        seen.add(normalized)
        rows = _edit_rows(runtime_module, item)
        action_text_bytes += sum(_byte_len(old) + _byte_len(new) for old, new in rows)
        if action_text_bytes > _MAX_ACTION_TEXT_BYTES:
            raise error(
                f"Source edit action exceeds {_MAX_ACTION_TEXT_BYTES} text bytes; "
                "apply the current localized subset and continue in another turn"
            )

        target = _regular_target(runtime_module, root, path, normalized)
        if not target.exists():
            if len(rows) != 1 or rows[0][0] != "":
                raise error(
                    f"New file {normalized} requires exactly one edit with empty old_text"
                )
            operations.append(
                {
                    "operation": "create",
                    "path": normalized,
                    "content": rows[0][1],
                }
            )
            continue

        original_bytes = target.read_bytes()
        try:
            content = original_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise error(f"Model source target is not UTF-8 text: {normalized}") from exc
        expected_sha256 = "sha256:" + hashlib.sha256(original_bytes).hexdigest()
        for old_text, new_text in rows:
            if not old_text:
                raise error(
                    f"Existing file {normalized} requires non-empty old_text; "
                    "retrieve a narrow current anchor first"
                )
            occurrences = content.count(old_text)
            if occurrences != 1:
                raise error(
                    f"Edit anchor for {normalized} matched {occurrences} times; "
                    "retrieve the current file/symbol and use one exact unique span"
                )
            content = content.replace(old_text, new_text, 1)

        operations.append(
            {
                "operation": "replace",
                "path": normalized,
                "expected_sha256": expected_sha256,
                "content": content,
            }
        )

    return {"project_root": project_root_argument, "operations": operations}


def install(runtime_module: Any) -> None:
    """Install one explicit model-facing ACI over the canonical host patch tool."""

    if bool(getattr(runtime_module, _MARKER, False)):
        return

    runtime_module._MODEL_SOURCE_PATCH_SCHEMA = BOUNDED_SOURCE_EDIT_SCHEMA

    def materialize(workspace_root: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
        return materialize_bounded_source_edit(runtime_module, workspace_root, payload)

    runtime_module._materialize_model_source_patch = materialize

    current_schemas = runtime_module.AgentToolRuntime.tool_schemas
    if not bool(getattr(current_schemas, _SCHEMA_MARKER, False)):
        @wraps(current_schemas)
        def tool_schemas(self: Any, stage: str):
            rows = current_schemas(self, stage)
            projected: list[dict[str, Any]] = []
            for row in rows:
                copied = dict(row)
                function = copied.get("function")
                if not isinstance(function, Mapping) or str(function.get("name", "")) != "apply_source_patch":
                    projected.append(copied)
                    continue
                patched_function = dict(function)
                patched_function["description"] = _MODEL_DESCRIPTION
                patched_function["parameters"] = BOUNDED_SOURCE_EDIT_SCHEMA
                copied["function"] = patched_function
                projected.append(copied)
            return tuple(projected)

        setattr(tool_schemas, _SCHEMA_MARKER, True)
        runtime_module.AgentToolRuntime.tool_schemas = tool_schemas

    setattr(runtime_module, _MARKER, True)


__all__ = ["BOUNDED_SOURCE_EDIT_SCHEMA", "install", "materialize_bounded_source_edit"]
