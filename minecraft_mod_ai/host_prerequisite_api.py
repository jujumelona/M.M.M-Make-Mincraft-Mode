"""Complete declaration evidence from materialized Java dependencies.

Source prefixes are not API contracts: an initializer can consume the entire
budget before a single usable method, or hide the fifth dependency entirely.
Tree-sitter removes implementation bodies rather than truncating declarations.
The project compiler remains the authority for type checking these APIs.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .java_region_parser import _parse, _text

_TYPES = frozenset({
    "class_declaration", "interface_declaration", "enum_declaration",
    "record_declaration", "annotation_type_declaration",
})


def _is_private(node) -> bool:
    modifiers = next((child for child in node.named_children if child.type == "modifiers"), None)
    return modifiers is not None and any(child.type == "private" for child in modifiers.children)


def _members(body):
    for child in body.named_children:
        if child.type == "enum_body_declarations":
            yield from child.named_children
        else:
            yield child


def java_api_declarations(source: str, *, path: str) -> dict:
    raw, root = _parse(source)
    package = next((
        _text(raw, identifier)
        for child in root.named_children if child.type == "package_declaration"
        for identifier in child.named_children
        if identifier.type in {"identifier", "scoped_identifier"}
    ), "")
    imports = [_text(raw, child) for child in root.named_children
               if child.type == "import_declaration"]
    declarations = []

    def visit(node, parent_owner: str):
        if _is_private(node):
            return
        name = node.child_by_field_name("name")
        body = node.child_by_field_name("body")
        if name is None or body is None:
            return
        owner = ".".join(part for part in (parent_owner, _text(raw, name)) if part)
        members = []
        for member in _members(body):
            if _is_private(member):
                continue
            if member.type in _TYPES:
                visit(member, owner)
            elif member.type in {"method_declaration", "constructor_declaration",
                                 "compact_constructor_declaration", "annotation_type_element_declaration"}:
                implementation = member.child_by_field_name("body")
                end = implementation.start_byte if implementation else member.end_byte
                header = raw[member.start_byte:end].decode("utf-8").strip().rstrip(";")
                if member.type == "compact_constructor_declaration":
                    parameters = node.child_by_field_name("parameters")
                    header += _text(raw, parameters)
                members.append(header + ";")
            elif member.type in {"field_declaration", "constant_declaration"}:
                field_type = member.child_by_field_name("type")
                if field_type is None:
                    continue
                prefix = raw[member.start_byte:field_type.end_byte].decode("utf-8")
                names = []
                for declarator in member.named_children:
                    if declarator.type != "variable_declarator":
                        continue
                    field_name = declarator.child_by_field_name("name")
                    if field_name is not None:
                        dimensions = declarator.child_by_field_name("dimensions")
                        names.append(_text(raw, field_name) + (
                            _text(raw, dimensions) if dimensions is not None else ""))
                if names:
                    members.append(prefix + " " + ", ".join(names) + ";")
            elif member.type == "enum_constant":
                enum_name = member.child_by_field_name("name")
                if enum_name is not None:
                    members.append(_text(raw, enum_name))
        declarations.append({
            "owner": owner,
            "declaration": raw[node.start_byte:body.start_byte].decode("utf-8").strip(),
            "members": members,
        })

    for child in root.named_children:
        if child.type in _TYPES:
            visit(child, package)
    return {
        "path": path, "authority": "materialized_prerequisite_declarations",
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "imports": imports, "types": declarations,
    }


def read_prerequisite_apis(context: dict, *, target_path: str, source_context=()) -> list[dict]:
    """Read all dependency APIs once, preserving overloads and generic types."""
    root = Path(context["project_root"]).resolve()
    rows, seen = [], set()
    for item in source_context:
        relative = item.get("path") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not relative.endswith(".java"):
            continue
        requested = root / relative
        absolute = requested.resolve()
        if (not absolute.is_relative_to(root) or not absolute.is_file()
                or requested.is_symlink() or relative == target_path or absolute in seen):
            continue
        seen.add(absolute)
        rows.append(java_api_declarations(absolute.read_text(encoding="utf-8"), path=relative))
    return rows
