from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .model_meta_output_contract import assert_design_field_clean
from .spec import SpecValidationError

_REQUIREMENT_ID_RE = re.compile(r"\breq_[A-Za-z0-9_]+\b")


def _referenced_requirement_ids(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, str):
        refs.update(_REQUIREMENT_ID_RE.findall(value))
    elif isinstance(value, Mapping):
        for key, child in value.items():
            refs.update(_referenced_requirement_ids(str(key)))
            refs.update(_referenced_requirement_ids(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            refs.update(_referenced_requirement_ids(child))
    return refs


def _assert_known_requirement_ids(
    field: str,
    value: Any,
    approved_requirement_ids: set[str],
) -> None:
    if not approved_requirement_ids:
        return
    unknown = sorted(_referenced_requirement_ids(value) - approved_requirement_ids)
    if unknown:
        raise SpecValidationError(
            f"{field} cites unknown requirement ids: " + ", ".join(unknown)
        )


def _active_requirement_ledger(prompt: str) -> tuple[dict[str, Any], ...]:
    """Read the already-frozen authored request authority without rebuilding scope."""
    from .planning_authority import active_authoritative_request_catalog

    catalog = active_authoritative_request_catalog(prompt)
    if not isinstance(catalog, Mapping):
        return ()
    raw_requirements = catalog.get("requirements", [])
    if not isinstance(raw_requirements, list):
        return ()
    ledger: list[dict[str, Any]] = []
    for raw in raw_requirements:
        if not isinstance(raw, Mapping):
            continue
        requirement_id = str(raw.get("requirement_id") or "").strip()
        if not requirement_id:
            continue
        span = raw.get("source_span")
        span_text = (
            str(span.get("text") or "").strip() if isinstance(span, Mapping) else ""
        )
        behavior = raw.get("observable_behavior")
        acceptance = raw.get("acceptance")
        ledger.append(
            {
                "requirement_id": requirement_id,
                "capability": str(raw.get("capability") or "").strip(),
                "authored_text": span_text or str(raw.get("statement") or "").strip(),
                "semantic_statement": str(raw.get("semantic_statement") or "").strip(),
                "observable_behavior": dict(behavior)
                if isinstance(behavior, Mapping)
                else {},
                "acceptance": [
                    str(item).strip() for item in acceptance if str(item).strip()
                ]
                if isinstance(acceptance, list)
                else [],
            }
        )
    return tuple(ledger)


def _render_requirement_ledger(ledger: Sequence[Mapping[str, Any]]) -> str:
    if not ledger:
        return "No approved requirement ledger is active."
    lines = ["APPROVED REQUIREMENTS (HOST AUTHORITY; preserve IDs exactly)"]
    for item in ledger:
        requirement_id = " ".join(str(item.get("requirement_id") or "").split())
        lines.append(f"- requirement_id: {requirement_id}")
        capability = " ".join(str(item.get("capability") or "").split())
        if capability:
            lines.append(f"  capability: {capability}")
        authored = " ".join(str(item.get("authored_text") or "").split())
        if authored:
            lines.append(f"  authored_text: {authored}")
        semantic = " ".join(str(item.get("semantic_statement") or "").split())
        if semantic:
            lines.append(f"  semantic_statement: {semantic}")
        acceptance = item.get("acceptance")
        if isinstance(acceptance, list):
            rendered = "; ".join(
                " ".join(str(value).split())
                for value in acceptance
                if str(value).strip()
            )
            if rendered:
                lines.append(f"  acceptance: {rendered}")
    return "\n".join(lines)


def _nonempty_text_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SpecValidationError(
            f"{field} must be a non-empty list; empty accepted design is forbidden"
        )
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    if len(cleaned) != len(value) or not cleaned:
        raise SpecValidationError(
            f"{field} must contain only non-empty authored design entries"
        )
    return cleaned


def _validate_section_types(
    section: Mapping[str, Any],
    fields: Sequence[str],
    *,
    requirement_ids: Sequence[str] = (),
) -> None:
    required_ids = {
        str(value).strip() for value in requirement_ids if str(value).strip()
    }
    for field in fields:
        if field not in section:
            raise SpecValidationError(f"section omitted required field {field!r}")
        value = section.get(field)
        if field in {"title", "pitch"}:
            if not isinstance(value, str) or not value.strip():
                raise SpecValidationError(f"{field} must be a non-empty string")
        elif field in {"core_loop", "progression", "acceptance_tests"}:
            _nonempty_text_list(value, field=field)
        elif field == "assets":
            if not isinstance(value, list):
                raise SpecValidationError("assets must be a list")
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise SpecValidationError(f"assets[{index}] must be an object")
                for key in ("id", "kind", "brief"):
                    if not str(item.get(key) or "").strip():
                        raise SpecValidationError(
                            f"assets[{index}].{key} must be non-empty"
                        )
        elif field == "modules":
            if not isinstance(value, list):
                raise SpecValidationError("modules must be a list")
            if required_ids and not value:
                raise SpecValidationError(
                    "modules must be non-empty while approved authored requirements exist"
                )
            for index, item in enumerate(value):
                if not isinstance(item, Mapping):
                    raise SpecValidationError(f"modules[{index}] must be an object")
                for key in ("plugin_id", "status", "reason"):
                    if not str(item.get(key) or "").strip():
                        raise SpecValidationError(
                            f"modules[{index}].{key} must be non-empty"
                        )
                _nonempty_text_list(
                    item.get("implementation_obligations"),
                    field=f"modules[{index}].implementation_obligations",
                )
                refs = item.get("requirement_refs")
                if not isinstance(refs, list):
                    raise SpecValidationError(
                        f"modules[{index}].requirement_refs must be a list"
                    )
                if required_ids:
                    refs = _nonempty_text_list(
                        refs, field=f"modules[{index}].requirement_refs"
                    )
                    unknown = sorted(set(refs) - required_ids)
                    if unknown:
                        raise SpecValidationError(
                            "module cites unknown requirement ids: "
                            + ", ".join(unknown)
                        )
        elif field in {"combat", "mod_context", "art_direction"}:
            if not isinstance(value, dict):
                raise SpecValidationError(f"{field} must be an object")
            if field in {"combat", "mod_context"}:
                for key, items in value.items():
                    if not str(key).strip():
                        raise SpecValidationError(f"{field} contains an empty key")
                    _nonempty_text_list(items, field=f"{field}.{key}")
        _assert_known_requirement_ids(field, value, required_ids)
        try:
            assert_design_field_clean(field, value)
        except ValueError as exc:
            raise SpecValidationError(str(exc)) from exc


def _validate_requirement_coverage(
    design: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required_ids = tuple(
        str(item.get("requirement_id") or "").strip()
        for item in ledger
        if str(item.get("requirement_id") or "").strip()
    )
    if not required_ids:
        return dict(design)
    known = set(required_ids)
    modules = design.get("modules")
    if not isinstance(modules, list) or not modules:
        raise SpecValidationError(
            "design readiness failed: approved requirements exist but modules are empty"
        )
    covered: set[str] = set()
    binding_rows = {
        requirement_id: {
            "requirement_id": requirement_id,
            "module_ids": [],
            "implementation_obligations": [],
        }
        for requirement_id in required_ids
    }
    for index, item in enumerate(modules):
        if not isinstance(item, Mapping):
            raise SpecValidationError(f"modules[{index}] must be an object")
        module_id = str(item.get("plugin_id") or "").strip()
        refs = _nonempty_text_list(
            item.get("requirement_refs"), field=f"modules[{index}].requirement_refs"
        )
        obligations = _nonempty_text_list(
            item.get("implementation_obligations"),
            field=f"modules[{index}].implementation_obligations",
        )
        unknown = sorted(set(refs) - known)
        if unknown:
            raise SpecValidationError(
                "design readiness failed: unknown requirement refs "
                + ", ".join(unknown)
            )
        for requirement_id in refs:
            covered.add(requirement_id)
            row = binding_rows[requirement_id]
            if module_id not in row["module_ids"]:
                row["module_ids"].append(module_id)
            for obligation in obligations:
                if obligation not in row["implementation_obligations"]:
                    row["implementation_obligations"].append(obligation)
    missing = [
        requirement_id
        for requirement_id in required_ids
        if requirement_id not in covered
    ]
    if missing:
        raise SpecValidationError(
            "design readiness failed: approved requirements have no implementation-bearing design module: "
            + ", ".join(missing)
        )
    result = dict(design)
    result["_requirement_design_bindings"] = {
        "schema_version": "mmm/requirement-design-binding-v1",
        "requirement_ids": list(required_ids),
        "bindings": [binding_rows[requirement_id] for requirement_id in required_ids],
    }
    return result
