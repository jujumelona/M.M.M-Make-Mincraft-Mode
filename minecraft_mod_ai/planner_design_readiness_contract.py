from __future__ import annotations

"""Host-owned design-readiness bootstrap checks.

Canonical schema and requirement validation live in their dedicated contract modules.
This bootstrap only verifies the live schema surface; it never reaches through retired
planning internals or recreates prompt-semantic authority.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .design_requirement_contract import (
    _active_requirement_ledger as _requirement_ledger,
    _nonempty_text_list as _require_nonempty_text_list,
    _validate_requirement_coverage,
    _validate_section_types,
)
from .design_section_schema import _SECTION_SPECS

_INSTALLED = False


def _active_requirement_ledger(prompt: str) -> tuple[dict[str, Any], ...]:
    return _requirement_ledger(prompt)


def _nonempty_text_list(value: Any, *, field: str) -> list[str]:
    return _require_nonempty_text_list(value, field=field)


def _strict_validate_section_types(
    section: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    ledger_ids = tuple(
        item["requirement_id"]
        for item in _active_requirement_ledger("")
        if str(item.get("requirement_id") or "").strip()
    )
    _validate_section_types(section, fields, requirement_ids=ledger_ids)


def _validate_design_coverage(
    design: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _validate_requirement_coverage(design, ledger)


def _assert_module_trace_schema() -> None:
    """Assert the canonical section schema contains implementation-bearing modules."""
    for section_id, _fields, properties in _SECTION_SPECS:
        if section_id != "modules_and_assets":
            continue
        modules = properties.get("modules")
        if not isinstance(modules, Mapping):
            raise RuntimeError("modules_and_assets schema lost its modules object")
        item_schema = modules.get("items")
        if not isinstance(item_schema, Mapping):
            raise RuntimeError("modules schema lost its item object")
        item_properties = item_schema.get("properties")
        required = item_schema.get("required")
        if not isinstance(item_properties, Mapping) or not isinstance(required, list):
            raise RuntimeError("modules schema is not contract-compatible")
        expected = {
            "plugin_id",
            "status",
            "reason",
            "requirement_refs",
            "implementation_obligations",
        }
        missing_properties = sorted(expected - set(item_properties))
        missing_required = sorted(expected - set(required))
        if missing_properties or missing_required:
            raise RuntimeError(
                "canonical modules schema is incomplete: "
                f"missing_properties={missing_properties} "
                f"missing_required={missing_required}"
            )
        return
    raise RuntimeError("modules_and_assets section schema was not found")


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _assert_module_trace_schema()
    _INSTALLED = True


__all__ = ["install"]
