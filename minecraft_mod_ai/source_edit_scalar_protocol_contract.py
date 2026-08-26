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
    "replace_exact": "replace_exact",
    "replace_all": "replace_exact",
    "create": "create_file",
    "create_file": "create_file",
    "create_class": "create_java_type",
    "create_type": "create_java_type",
    "create_java_type": "create_java_type",
    "create_java_class": "create_java_type",
    "delete": "delete_file",
    "delete_file": "delete_file",
    "remove": "delete_file",
    "remove_file": "delete_file",
    "apply_source_edit": "replace_exact",
    "apply_source_patch": "replace_exact",
    "edit": "replace_exact",
    "write": "create_file",
    "write_file": "create_file",
    "modify": "replace_exact",
    "patch": "replace_exact",
    "patch_file": "replace_exact",
    "update": "replace_exact",
    "insert": "insert_before",
    "insert_before": "insert_before",
    "insert_after": "insert_after",
    "append": "insert_after",
    "prepend": "insert_before",
    "add_import": "add_java_import",
    "add_java_import": "add_java_import",
    "import": "add_java_import",
    "insert_member": "insert_java_member",
    "insert_java_member": "insert_java_member",
    "add_member": "insert_java_member",
    "add_method": "insert_java_member",
    "add_field": "insert_java_member",
}
_MODEL_OPERATION_ENUM = (*_CANONICAL_OPERATIONS, "replace", "create", "delete")
SOURCE_EDIT_PARAMETER_ALIASES = {
    "file": "path",
    "target_path": "path",
    "target_file": "path",
    "new_text": "new",
    "new_content": "new",
    "replacement": "new",
    "old_text": "old",
    "code": "content",
    "body": "content",
}
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
            "enum": list(_MODEL_OPERATION_ENUM),
            "description": (
                "Perform exactly one semantic source action. For Java: create_java_type "
                "creates only the empty type shell, add_java_import adds one import, and "
                "insert_java_member adds one field/constructor/method/nested declaration. "
                "Never emit an entire Java file as create_file content."
            ),
        },
        "path": {
            "type": "string",
            "description": "Workspace-relative file path (for example src/main/java/...)",
        },
        "old": {
            "type": "string",
            "description": "Exact text to match for replace_exact",
        },
        "new": {
            "type": "string",
            "description": "Replacement text for replace_exact",
        },
        "anchor": {
            "type": "string",
            "description": "Exact anchor text for insert_before or insert_after",
        },
        "content": {
            "type": "string",
            "description": "Text to insert (insert_before/insert_after) or create_file content",
        },
        "text": {
            "type": "string",
            "description": (
                "Convenience alias for new (replace_exact), member (insert_java_member), "
                "or content (insert_before/insert_after/create_file)"
            ),
        },
        "count": {
            "type": "integer",
            "description": "Must be 1 when provided",
        },
        "package_name": {
            "type": "string",
            "description": "Java package name for create_java_type",
        },
        "declaration": {
            "type": "string",
            "description": "Type header without braces/body for create_java_type",
        },
        "import_name": {
            "type": "string",
            "description": "Fully-qualified class or static member import name for add_java_import",
        },
        "member": {
            "type": "string",
            "description": "One Java member declaration with braces/body for insert_java_member",
        },
        "new_text": {
            "type": "string",
            "description": "Alias for new in replace_exact",
        },
        "new_content": {
            "type": "string",
            "description": "Alias for new / content",
        },
        "replacement": {
            "type": "string",
            "description": "Alias for new in replace_exact",
        },
        "old_text": {
            "type": "string",
            "description": "Alias for old in replace_exact",
        },
        "code": {
            "type": "string",
            "description": "Alias for content / member",
        },
        "body": {
            "type": "string",
            "description": "Alias for member / content",
        },
        "file": {
            "type": "string",
            "description": "Alias for path",
        },
        "target_path": {
            "type": "string",
            "description": "Alias for path",
        },
        "target_file": {
            "type": "string",
            "description": "Alias for path",
        },
    },
}
_ALLOWED_FIELDS = frozenset(SOURCE_EDIT_SCHEMA["properties"].keys())


def _required_text(
    runtime_module: Any,
    payload: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise runtime_module.AgentToolRuntimeError(f"Field {key!r} must be a non-empty string")
    return value


def _canonicalize_payload_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for alias, canonical in SOURCE_EDIT_PARAMETER_ALIASES.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
    return normalized


def _normalize_operation(runtime_module: Any, value: Any, payload: Mapping[str, Any] | None = None) -> str:
    if not isinstance(value, str):
        raise runtime_module.AgentToolRuntimeError("operation must be a string")
    clean = value.strip()
    if clean in ("apply_source_edit", "apply_source_patch", "edit", "modify", "patch", "update", "write", "create", "insert") and payload is not None:
        if payload.get("old") or payload.get("old_text"):
            return "replace_exact"
        if payload.get("anchor"):
            return "insert_before"
        if payload.get("member"):
            return "insert_java_member"
        if payload.get("import_name"):
            return "add_java_import"
        if payload.get("package_name") or payload.get("declaration"):
            return "create_java_type"
        if payload.get("content") or payload.get("text") or payload.get("code") or payload.get("new") or payload.get("new_text"):
            return "create_file"

    operation = _OPERATION_ALIASES.get(clean, clean)
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
        package_match = re.search(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*", package_name)
        if package_match:
            package_name = package_match.group(0)
        else:
            raise runtime_module.AgentToolRuntimeError(
                f"Invalid Java package_name: {package_name!r}"
            )
    declaration = " ".join(
        _required_text(runtime_module, payload, "declaration").strip().split()
    )
    declaration = declaration.split("{")[0].rstrip().rstrip(";").strip()
    if not _JAVA_TYPE_DECLARATION.search(declaration):
        declaration = f"public class {declaration}"

    content = f"package {package_name};\n\n{declaration} {{\n}}\n"
    return {
        "operation": "create",
        "path": normalized,
        "content": content,
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
    if import_name.startswith("import "):
        import_name = import_name[7:].strip()
    import_name = import_name.rstrip(";").strip()
    statement = f"import {import_name};"
    if re.search(rf"(?m)^\s*{re.escape(statement)}\s*$", text):
        return _host_replace(normalized, expected_sha256, text)
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
    extra_imports: list[str] = []
    clean_member_lines: list[str] = []
    for line in member.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("import "):
            extra_imports.append(trimmed if trimmed.endswith(";") else trimmed + ";")
        elif trimmed.startswith("package "):
            continue
        else:
            clean_member_lines.append(line)

    member_code = "\n".join(clean_member_lines).strip()
    current_text = text
    for imp in extra_imports:
        if not re.search(rf"(?m)^\s*{re.escape(imp)}\s*$", current_text):
            pkg = re.search(r"(?m)^\s*package\s+[\w.$]+\s*;\s*$", current_text)
            existing_imports = list(re.finditer(r"(?m)^\s*import\s+[^;\n]+;\s*$", current_text))
            if existing_imports:
                ins_pos = existing_imports[-1].end()
                current_text = current_text[:ins_pos] + "\n" + imp + current_text[ins_pos:]
            elif pkg:
                ins_pos = pkg.end()
                current_text = current_text[:ins_pos] + "\n\n" + imp + current_text[ins_pos:]
            else:
                current_text = imp + "\n\n" + current_text

    close_at = _outer_type_close(runtime_module, current_text, normalized)
    prefix = current_text[:close_at]
    suffix = current_text[close_at:]
    formatted_lines = []
    for line in member_code.splitlines():
        formatted_lines.append(("    " + line) if line else "")
    formatted = "\n".join(formatted_lines).rstrip()
    if not formatted:
        return _host_replace(normalized, expected_sha256, current_text)
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

    payload = _canonicalize_payload_aliases(payload)
    operation = _normalize_operation(runtime_module, payload.get("operation"), payload)
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
        content = _required_text(runtime_module, payload, "content", allow_empty=True)
        if target.exists():
            raw_bytes, text, expected_sha256 = _read_utf8(runtime_module, target, normalized)
            del raw_bytes, text
            return {
                "project_root": project_root_argument,
                "operations": [
                    {"operation": "replace", "path": normalized, "expected_sha256": expected_sha256, "content": content}
                ],
            }
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

    if operation == "replace_exact" and "old" not in payload and "old_text" not in payload:
        new_text = str(payload.get("new") or payload.get("new_text") or payload.get("text") or "")
        return {
            "project_root": project_root_argument,
            "operations": [
                {
                    "operation": "replace",
                    "path": normalized,
                    "expected_sha256": expected_sha256,
                    "content": new_text,
                }
            ],
        }

    replacement = _replacement_for_edit(
        runtime_module,
        operation=operation,
        payload=payload,
        count=count,
    )
    norm_text = text.replace("\r\n", "\n")
    norm_old = replacement["old"].replace("\r\n", "\n")
    if norm_old != replacement["old"] and norm_old in norm_text:
        replacement["old"] = norm_old
        replacement["new"] = replacement["new"].replace("\r\n", "\n")
        text = norm_text

    found = text.count(replacement["old"])
    if found == 0:
        clean_lines = [line_str.strip() for line_str in norm_old.splitlines() if line_str.strip()]
        if clean_lines:
            file_lines = norm_text.splitlines()
            fuzzy_matches = [
                "\n".join(file_lines[i : i + len(clean_lines)])
                for i in range(len(file_lines) - len(clean_lines) + 1)
                if all(
                    file_lines[i + j].strip() == clean_lines[j]
                    for j in range(len(clean_lines))
                )
            ]
            found = len(fuzzy_matches)
            if found == 1:
                replacement["old"] = fuzzy_matches[0]
                text = norm_text

    if found == 0 and (not norm_old.strip() or norm_old == norm_text):
        return {
            "project_root": project_root_argument,
            "operations": [
                {
                    "operation": "replace",
                    "path": normalized,
                    "expected_sha256": expected_sha256,
                    "content": replacement["new"],
                }
            ],
        }

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


__all__ = ["SOURCE_EDIT_PARAMETER_ALIASES", "SOURCE_EDIT_SCHEMA", "materialize_model_source_edit"]
