from __future__ import annotations

"""Compact, deterministic catalog of canonical contract type shapes.

MMM already keeps many subsystem contracts in ``*_contract.py`` / ``*_contracts.py``
modules. Small coding models should not have to guess which similarly named payload
shape a producer or consumer expects. This module derives a compact manifest directly
from those canonical source files so the model context and the Python source cannot drift.
"""

import ast
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONTRACT_SUFFIXES = ("_contract.py", "_contracts.py")
_EXPLICIT_CONTRACT_FILES = frozenset({"target_contract.py"})
_IGNORED_CLASS_NAMES = frozenset({"Any"})


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


def _module_contracts(path: Path, package_root: Path) -> dict[str, Any]:
    relative = path.relative_to(package_root.parent).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=relative)
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
            types.append(
                {
                    "name": node.name,
                    "bases": tuple(
                        base
                        for base in (_base_name(base_node) for base_node in node.bases)
                        if base
                    ),
                    "fields": _class_fields(node),
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


@lru_cache(maxsize=1)
def contract_schema_manifest() -> tuple[dict[str, Any], ...]:
    """Return every canonical contract module and its explicit type shape.

    The manifest is sorted by repository-relative path so prompts, traces, and tests stay
    deterministic. It intentionally contains type names and annotated fields rather than
    full source text to keep small-model context compact.
    """

    package_root = Path(__file__).resolve().parent
    paths = sorted(
        path
        for path in package_root.rglob("*.py")
        if path.is_file() and _is_contract_file(path)
    )
    return tuple(_module_contracts(path, package_root) for path in paths)


__all__ = ["contract_schema_manifest"]
