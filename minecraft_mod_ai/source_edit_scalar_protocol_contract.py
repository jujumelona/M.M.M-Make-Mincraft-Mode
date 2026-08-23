from __future__ import annotations

"""Canonical model-facing source edit protocol.

One model action describes one executable semantic edit. Java source is never created
as one giant model-authored file payload: the model creates a type shell, adds imports,
and inserts one member per action. The host materializes those semantic actions into
SHA-bound transactional patches. Non-Java resources may still be created directly.
"""

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


_CANONICAL_OPERATIONS = (
    "replace_exact",
    "insert_before",
    "insert_after",
    "create_file",
    "create_java_type",
    "add_java_import",
    "insert_java_member",
    "delete_file",
)
_OPERATION_ALIASES = {
    "replace": "replace_exact",
    "create": "create_file",
    "delete": "delete_file",
}
_ACCEPTED_OPERATIONS = (*_CANONICAL_OPERATIONS, *_OPERATION_ALIASES)
_JAVA_PATH_SUFFIX = ".java"
_JAVA_PACKAGE = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*$")
_JAVA_TYPE_DECLARATION = re.compile(
    r"\b(?:class|interface|enum|record|@interface)\s+[A-Za-z_$][\w$]*\b"
)
_JAVA_TYPE_OPEN = re.compile(
    r"\b(?:class|interface|enum|record|@interface)\s+[A-Za-z_$][\w$]*[^;{]*\{"
)

SOURCE_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation", "path"],
    "properties": {
        "operation": {
            "type": "string",
            "enum": list(_ACCEPTED_OPERATIONS),
            "description": (
                "Perform exactly one semantic source action. For Java: create_java_type "
                "creates only the empty type shell, add_java_import adds one import, and "
                "insert_java_member adds one field/constructor/method/nested declaration. "
                "Never emit an entire Java file as create_file content."
            ),
        },
        "path": {
            "type": "string",
            "minLength": 1,
            "description": "Project-relative source/resource path selected for this action.",
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
            "description": (
                "Inserted text for anchor edits, or complete content only for a new "
                "non-Java resource. Java files must use structural Java operations."
            ),
        },
        "text": {
            "type": "string",
            "description": "Lossless Qwen alias for new/content/member.",
        },
        "count": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "Expected exact occurrence count for the selected span/anchor.",
        },
        "package_name": {
            "type": "string",
            "minLength": 1,
            "description": "Java package for create_java_type, e.g. com.example.mod.",
        },
        "declaration": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Java type header only, without braces or body, e.g. "
                "'public final class Example implements ModInitializer'."
            ),
        },
        "import_name": {
            "type": "string",
            "minLength": 1,
            "description": (
                "One Java import target without 'import' or trailing ';'. Prefix with "
                "'static ' for a static import."
            ),
        },
        "member": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Exactly one Java type member: one field, constructor, method, initializer, "
                "or nested type. Do not include package/import declarations or the outer type."
            ),
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
    if operation == "replace_exact":
        canonical_key = "new"
    elif operation == "insert_java_member":
        canonical_key = "member"
    else:
        canonical_key = "content"
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


def _read_utf8(runtime_module: Any, target: Path, normalized: str) -> tuple[bytes, str, str]:
    raw = target.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise runtime_module.AgentToolRuntimeError(
            f"Source edit target is not UTF-8 text: {normalized}"
        ) from exc
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    return raw, text, digest


def _host_replace(path: str, expected_sha256: str, content: str) -> dict[str, Any]:
    return {
        "operation": "replace",
        "path": path,
        "expected_sha256": expected_sha256,
        "content": content,
    }


def _validate_java_path(runtime_module: Any, normalized: str) -> None:
    if not normalized.endswith(_JAVA_PATH_SUFFIX):
        raise runtime_module.AgentToolRuntimeError(
            f"Java structural operation requires a .java path: {normalized}"
        )


def _create_java_type(
    runtime_module: Any,
    *,
    normalized: str,
    target: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_java_path(runtime_module, normalized)
    if target.exists():
        raise runtime_module.AgentToolRuntimeError(
            f"create_java_type target already exists: {normalized}"
        )
    package_name = _required_text(runtime_module, payload, "package_name").strip()
    if not _JAVA_PACKAGE.fullmatch(package_name):
        raise runtime_module.AgentToolRuntimeError(
            f"Invalid Java package_name: {package_name!r}"
        )
    declaration = " ".join(
        _required_text(runtime_module, payload, "declaration").strip().split()
    )
    if "{" in declaration or "}" in declaration or ";" in declaration:
        raise runtime_module.AgentToolRuntimeError(
            "create_java_type declaration must be a type header without braces/body"
        )
    if not _JAVA_TYPE_DECLARATION.search(declaration):
        raise runtime_module.AgentToolRuntimeError(
            "create_java_type declaration must declare one class/interface/enum/record"
        )
    return {
        "operation": "create",
        "path": normalized,
        "content": f"package {package_name};\n\n{declaration} {{\n}}\n",
    }


def _mask_java_noncode(text: str) -> str:
    chars = list(text)
    state = "code"
    escape = False
    index = 0
    while index < len(chars):
        char = chars[index]
        nxt = chars[index + 1] if index + 1 < len(chars) else ""
        if state == "code":
            if char == "/" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                state = "line_comment"
                index += 2
                continue
            if char == "/" and nxt == "*":
                chars[index] = chars[index + 1] = " "
                state = "block_comment"
                index += 2
                continue
            if char == '"':
                chars[index] = " "
                state = "string"
                escape = False
            elif char == "'":
                chars[index] = " "
                state = "char"
                escape = False
        elif state == "line_comment":
            if char == "\n":
                state = "code"
            else:
                chars[index] = " "
        elif state == "block_comment":
            if char == "*" and nxt == "/":
                chars[index] = chars[index + 1] = " "
                state = "code"
                index += 2
                continue
            if char != "\n":
                chars[index] = " "
        else:
            if char == "\n":
                chars[index] = " "
            else:
                chars[index] = " "
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif (state == "string" and char == '"') or (
                state == "char" and char == "'"
            ):
                state = "code"
        index += 1
    return "".join(chars)


def _outer_type_close(runtime_module: Any, text: str, normalized: str) -> int:
    masked = _mask_java_noncode(text)
    match = _JAVA_TYPE_OPEN.search(masked)
    if match is None:
        raise runtime_module.AgentToolRuntimeError(
            f"Could not find an outer Java type declaration in {normalized}"
        )
    opening = masked.find("{", match.start(), match.end())
    depth = 0
    for index in range(opening, len(masked)):
        char = masked[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise runtime_module.AgentToolRuntimeError(
        f"Outer Java type has unmatched braces in {normalized}"
    )


def _add_java_import(
    runtime_module: Any,
    *,
    normalized: str,
    text: str,
    expected_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_java_path(runtime_module, normalized)
    import_name = _required_text(runtime_module, payload, "import_name").strip()
    if import_name.startswith("import ") or import_name.endswith(";") or "\n" in import_name:
        raise runtime_module.AgentToolRuntimeError(
            "import_name must omit the 'import' keyword, semicolon, and newlines"
        )
    statement = f"import {import_name};"
    if re.search(rf"(?m)^\s*{re.escape(statement)}\s*$", text):
        raise runtime_module.AgentToolRuntimeError(
            f"Java import already exists in {normalized}: {import_name}"
        )
    package = re.search(r"(?m)^\s*package\s+[\w.$]+\s*;\s*$", text)
    imports = list(re.finditer(r"(?m)^\s*import\s+[^;\n]+;\s*$", text))
    if imports:
        insert_at = imports[-1].end()
        insertion = "\n" + statement
    elif package:
        insert_at = package.end()
        insertion = "\n\n" + statement
    else:
        insert_at = 0
        insertion = statement + "\n\n"
    updated = text[:insert_at] + insertion + text[insert_at:]
    return _host_replace(normalized, expected_sha256, updated)


def _insert_java_member(
    runtime_module: Any,
    *,
    normalized: str,
    text: str,
    expected_sha256: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_java_path(runtime_module, normalized)
    member = _required_text(runtime_module, payload, "member").strip()
    if re.search(r"(?m)^\s*(?:package|import)\s+", member):
        raise runtime_module.AgentToolRuntimeError(
            "insert_java_member may not contain package/import declarations"
        )
    close_at = _outer_type_close(runtime_module, text, normalized)
    prefix = text[:close_at]
    suffix = text[close_at:]
    formatted_lines = []
    for line in member.splitlines():
        formatted_lines.append(("    " + line) if line else "")
    formatted = "\n".join(formatted_lines).rstrip()
    if not formatted:
        raise runtime_module.AgentToolRuntimeError("member must contain Java source")
    spacer = "" if prefix.endswith("\n\n") else "\n" if prefix.endswith("\n") else "\n\n"
    updated = prefix + spacer + formatted + "\n" + suffix
    return _host_replace(normalized, expected_sha256, updated)


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
    requires_existing = operation not in {"create_file", "create_java_type"}
    normalized, target = _normalize_model_target(
        runtime_module,
        root,
        path,
        require_existing=requires_existing,
    )

    if operation == "create_java_type":
        forbidden = {"old", "new", "anchor", "content", "text", "count", "import_name", "member"}.intersection(payload)
        if forbidden:
            raise runtime_module.AgentToolRuntimeError(
                f"Fields {sorted(forbidden)} are invalid for create_java_type"
            )
        return {
            "project_root": project_root_argument,
            "operations": [
                _create_java_type(
                    runtime_module,
                    normalized=normalized,
                    target=target,
                    payload=payload,
                )
            ],
        }

    if operation == "create_file":
        forbidden = {"old", "new", "anchor", "count", "package_name", "declaration", "import_name", "member"}.intersection(payload)
        if forbidden:
            raise runtime_module.AgentToolRuntimeError(
                f"Fields {sorted(forbidden)} are invalid for create_file"
            )
        if normalized.endswith(_JAVA_PATH_SUFFIX):
            raise runtime_module.AgentToolRuntimeError(
                "Java files cannot be created as one whole-file payload; use "
                "create_java_type, then add_java_import / insert_java_member actions"
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

    raw_bytes, text, expected_sha256 = _read_utf8(runtime_module, target, normalized)
    del raw_bytes

    if operation == "add_java_import":
        forbidden = {"old", "new", "anchor", "content", "text", "count", "package_name", "declaration", "member"}.intersection(payload)
        if forbidden:
            raise runtime_module.AgentToolRuntimeError(
                f"Fields {sorted(forbidden)} are invalid for add_java_import"
            )
        return {
            "project_root": project_root_argument,
            "operations": [
                _add_java_import(
                    runtime_module,
                    normalized=normalized,
                    text=text,
                    expected_sha256=expected_sha256,
                    payload=payload,
                )
            ],
        }

    if operation == "insert_java_member":
        forbidden = {"old", "new", "anchor", "content", "count", "package_name", "declaration", "import_name"}.intersection(payload)
        if forbidden:
            raise runtime_module.AgentToolRuntimeError(
                f"Fields {sorted(forbidden)} are invalid for insert_java_member"
            )
        return {
            "project_root": project_root_argument,
            "operations": [
                _insert_java_member(
                    runtime_module,
                    normalized=normalized,
                    text=text,
                    expected_sha256=expected_sha256,
                    payload=payload,
                )
            ],
        }

    if operation == "delete_file":
        forbidden = {"old", "new", "anchor", "content", "text", "count", "package_name", "declaration", "import_name", "member"}.intersection(payload)
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

    structural_fields = {"package_name", "declaration", "import_name", "member"}.intersection(payload)
    if structural_fields:
        raise runtime_module.AgentToolRuntimeError(
            f"Fields {sorted(structural_fields)} are invalid for {operation}"
        )
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
        payload=payload,
        count=count,
    )
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
