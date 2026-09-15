from __future__ import annotations

"""Compact deterministic catalog and ownership audit for structured contracts.

MMM has many subsystem contracts. Small coding models and host code must not guess which
similarly named payload shape a producer or consumer expects. Type shapes are derived from
canonical contract modules, while schema identifiers are audited across the complete
runtime package so one versioned ``mmm/...-vN`` literal has exactly one Python owner.
Consumers must import the owner's symbol instead of redeclaring the same identifier.
"""

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONTRACT_SUFFIXES = ("_contract.py", "_contracts.py")
_EXPLICIT_CONTRACT_FILES = frozenset({"target_contract.py"})
_IGNORED_CLASS_NAMES = frozenset({"Any"})
_SCHEMA_ID_RE = re.compile(r"^mmm/[A-Za-z0-9._/-]+-v[0-9]+$")


def _is_contract_file(path: Path) -> bool:
    name = path.name
    return name in _EXPLICIT_CONTRACT_FILES or name.endswith(_CONTRACT_SUFFIXES)


def _annotation_text(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError):
        return ""


def _base_name(node: ast.AST) -> str:
    return _annotation_text(node)


def _class_fields(node: ast.ClassDef) -> tuple[dict[str, str], ...]:
    fields: list[dict[str, str]] = []
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
            continue
        fields.append(
            {
                "name": item.target.id,
                "type": _annotation_text(item.annotation),
                "required": "false" if item.value is not None else "true",
            }
        )
    return tuple(fields)


def _parse_module(path: Path, relative: str) -> ast.Module:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source, filename=relative)


def _module_contracts(path: Path, package_root: Path) -> dict[str, Any]:
    relative = path.relative_to(package_root.parent).as_posix()
    try:
        module = _parse_module(path, relative)
    except (OSError, UnicodeError, SyntaxError) as exc:
        return {
            "path": relative,
            "types": (),
            "parse_error": type(exc).__name__,
        }

    types: list[dict[str, Any]] = []
    aliases: list[dict[str, str]] = []
    for node in module.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            if node.name in _IGNORED_CLASS_NAMES:
                continue
            fields = _class_fields(node)
            if not fields:
                continue
            types.append(
                {
                    "name": node.name,
                    "bases": tuple(
                        base
                        for base in (_base_name(base_node) for base_node in node.bases)
                        if base
                    ),
                    "fields": fields,
                }
            )
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotation = _annotation_text(node.annotation)
            if "TypeAlias" in annotation:
                aliases.append(
                    {
                        "name": node.target.id,
                        "type": _annotation_text(node.value),
                    }
                )

    return {
        "path": relative,
        "types": tuple(types),
        "aliases": tuple(aliases),
    }


def _schema_assignment_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Assign):
        return tuple(
            target.id
            for target in node.targets
            if isinstance(target, ast.Name) and "SCHEMA" in target.id.upper()
        )
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and "SCHEMA" in node.target.id.upper()
    ):
        return (node.target.id,)
    return ()


def _schema_assignment_value(node: ast.AST) -> str:
    value: ast.AST | None
    if isinstance(node, ast.Assign):
        value = node.value
    elif isinstance(node, ast.AnnAssign):
        value = node.value
    else:
        value = None
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        text = value.value.strip()
        return text if _SCHEMA_ID_RE.fullmatch(text) else ""
    return ""


@lru_cache(maxsize=1)
def contract_schema_manifest() -> tuple[dict[str, Any], ...]:
    """Return every canonical contract module and its explicit payload type shape."""

    package_root = Path(__file__).resolve().parent
    paths = sorted(
        path
        for path in package_root.rglob("*.py")
        if path.is_file() and _is_contract_file(path)
    )
    return tuple(_module_contracts(path, package_root) for path in paths)


@lru_cache(maxsize=1)
def contract_schema_literal_manifest() -> tuple[dict[str, str], ...]:
    """Return every direct runtime schema-ID declaration in the package.

    Only module-level assignments with ``SCHEMA`` in the symbol name and a versioned
    ``mmm/...-vN`` string are owners. Imported aliases therefore do not create a second
    owner, which is exactly the desired compatibility-module behavior.
    """

    package_root = Path(__file__).resolve().parent
    records: list[dict[str, str]] = []
    for path in sorted(package_root.rglob("*.py")):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root.parent).as_posix()
        try:
            module = _parse_module(path, relative)
        except (OSError, UnicodeError, SyntaxError):
            continue
        for node in module.body:
            schema_id = _schema_assignment_value(node)
            if not schema_id:
                continue
            for name in _schema_assignment_names(node):
                records.append(
                    {
                        "schema_id": schema_id,
                        "path": relative,
                        "symbol": name,
                    }
                )
    return tuple(records)


def duplicate_contract_schema_ids() -> dict[str, tuple[dict[str, str], ...]]:
    """Return versioned schema IDs with more than one direct runtime owner."""

    grouped: dict[str, list[dict[str, str]]] = {}
    for record in contract_schema_literal_manifest():
        grouped.setdefault(record["schema_id"], []).append(record)
    return {
        schema_id: tuple(records)
        for schema_id, records in sorted(grouped.items())
        if len(records) > 1
    }


__all__ = [
    "contract_schema_literal_manifest",
    "contract_schema_manifest",
    "duplicate_contract_schema_ids",
]
