from __future__ import annotations

"""Canonical project-path contract for generation receipts.

Generation receipts contain identifiers, semantic observations, evidence, ports, and
filesystem mutations. Only explicit receipt fields may claim project paths. Callers
must not recursively reinterpret arbitrary files/path keys as filesystem outputs.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_FILE_LIST_SCHEMAS = {
    "mmm/extended-content-v2",
    "mmm/system-pack-generation-v5",
    "mmm/geckolib-generation-v3",
    "mmm/local-ai-sidecar-generation-v1",
}
_TOUCH_LIST_SCHEMAS = {
    "mmm/extended-content-v2",
    "mmm/artifact-graph-execution-receipt-v1",
    "mmm/custom-module-result-v3",
}
_CHILD_RECEIPT_KEYS = {
    "source_receipt",
    "binding_receipt",
    "write_receipt",
    "patch_receipt",
    "patch",
}
_MUTATING_OPERATIONS = {"create", "replace", "edit", "delete"}
_OUTPUT_OPERATIONS = {"create", "replace", "edit"}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    return ()


def _path_strings(value: Any) -> tuple[str, ...]:
    result: list[str] = []
    for item in _sequence(value):
        if isinstance(item, (str, Path)):
            rendered = str(item).strip()
            if rendered:
                result.append(rendered)
    return tuple(result)


def _add(target: list[str], seen: set[str], value: Any) -> None:
    if not isinstance(value, (str, Path)):
        return
    rendered = str(value).strip()
    if rendered and rendered not in seen:
        seen.add(rendered)
        target.append(rendered)


def _collect(receipt: Any, *, include_deleted: bool) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(value: Any) -> None:
        _add(ordered, seen, value)

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return

        schema = value.get("schema_version")
        schema = schema if isinstance(schema, str) else ""

        operation = value.get("operation")
        path = value.get("path")
        allowed_operations = (
            _MUTATING_OPERATIONS if include_deleted else _OUTPUT_OPERATIONS
        )
        if operation in allowed_operations and isinstance(path, str):
            add(path)

        if schema == "mmm/source-patch-receipt-v1":
            for item in _sequence(value.get("operations")):
                if isinstance(item, Mapping):
                    visit(item)
            return

        if schema == "mmm/generation-work-node-v1":
            for child in _sequence(value.get("receipts")):
                if isinstance(child, Mapping):
                    visit(child)
            return

        if schema in _TOUCH_LIST_SCHEMAS:
            for raw in _path_strings(value.get("touched_paths")):
                add(raw)

        if schema in _FILE_LIST_SCHEMAS:
            for raw in _path_strings(value.get("files")):
                add(raw)

        if schema == "mmm/research-ledger-write-receipt-v1":
            add(value.get("target_path"))

        if schema == "mmm/resource-production-receipt-v2":
            for item in _sequence(value.get("assets")):
                if isinstance(item, Mapping) and item.get("container") == "mod":
                    add(item.get("target"))
            for item in _sequence(value.get("documents")):
                if isinstance(item, Mapping) and item.get("container") == "mod":
                    add(item.get("resolved_path"))

        if include_deleted:
            for key in ("deleted_files", "removed_files"):
                for raw in _path_strings(value.get(key)):
                    add(raw)

        for key in ("touched_paths", "written_files", "generated_files"):
            if key == "touched_paths" and schema in _TOUCH_LIST_SCHEMAS:
                continue
            for raw in _path_strings(value.get(key)):
                add(raw)

        for key in _CHILD_RECEIPT_KEYS:
            child = value.get(key)
            if isinstance(child, Mapping):
                visit(child)
            else:
                for item in _sequence(child):
                    if isinstance(item, Mapping):
                        visit(item)

        nested_receipts = value.get("receipts")
        if isinstance(nested_receipts, Mapping):
            for child in nested_receipts.values():
                if isinstance(child, Mapping):
                    visit(child)
        elif schema != "mmm/generation-work-node-v1":
            for child in _sequence(nested_receipts):
                if isinstance(child, Mapping):
                    visit(child)

    visit(receipt)
    return tuple(ordered)


def receipt_mutation_paths(receipt: Any) -> tuple[str, ...]:
    """Return project paths whose bytes or existence may have changed."""

    return _collect(receipt, include_deleted=True)


def receipt_output_paths(receipt: Any) -> tuple[str, ...]:
    """Return project paths that must exist after a successful generation receipt."""

    return _collect(receipt, include_deleted=False)


__all__ = ["receipt_mutation_paths", "receipt_output_paths"]
