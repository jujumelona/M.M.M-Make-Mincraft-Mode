from __future__ import annotations

"""Single source of truth for authored atomic production contracts.

The implementation-IR merger, graph-to-task binder, and atomic Java consumer all
use this module. No producer or consumer owns an independent serialization format.
"""

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

AUTHORED_ATOMIC_CONTRACT_SCHEMA_VERSION = "mmm/authored-atomic-contract-v1"


def _requirement_sort_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
    key = str(item[0] or "")
    match = re.fullmatch(r"R(\d+)", key)
    if match:
        return (0, int(match.group(1)))
    return (1, key)


def decode_atomic_obligation(
    raw: Any,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
        instruction = json.loads(str(payload.get("instruction") or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(instruction, dict):
        return None
    return payload, instruction


def authored_obligation_record(
    raw: Any,
) -> tuple[tuple[str, str], dict[str, Any], dict[str, Any]] | None:
    decoded = decode_atomic_obligation(raw)
    if decoded is None:
        return None
    payload, instruction = decoded
    section = str(instruction.get("section") or "").strip()
    concern = str(
        instruction.get("concern_template")
        or instruction.get("concern")
        or ""
    ).strip()
    if not section or not concern:
        return None
    return (section, concern), payload, instruction


def merge_authored_obligation_value(existing: str, proposed: str) -> str | None:
    left = authored_obligation_record(existing)
    right = authored_obligation_record(proposed)
    if left is None or right is None or left[0] != right[0]:
        return None
    _identity, left_payload, left_instruction = left
    _right_identity, right_payload, right_instruction = right
    if left_instruction != right_instruction:
        return None

    left_sources = left_payload.get("source_requirements")
    right_sources = right_payload.get("source_requirements")
    if not isinstance(left_sources, dict) or not isinstance(right_sources, dict):
        return None

    merged_sources = deepcopy(left_sources)
    changed = False
    for requirement_id, source_text in right_sources.items():
        if requirement_id in merged_sources:
            if merged_sources[requirement_id] != source_text:
                raise ValueError(
                    "IMPLEMENTATION_IR_SOURCE_REQUIREMENT_CONFLICT: "
                    + str(requirement_id)
                )
            continue
        merged_sources[requirement_id] = source_text
        changed = True

    if not changed:
        return existing
    merged_payload = deepcopy(left_payload)
    merged_payload["source_requirements"] = merged_sources
    return json.dumps(merged_payload, ensure_ascii=False, sort_keys=True)


def concern_source_requirements(
    requirements: Mapping[str, Any],
    *,
    concern: str,
) -> dict[str, str]:
    """Resolve one concern from either canonical heading or legacy bullet syntax."""

    from .authored_ir_parser import (
        parse_markdown_heading,
        section_slug,
        slice_concern_requirements,
    )

    ordered = [
        (str(key), str(value))
        for key, value in sorted(
            dict(requirements or {}).items(),
            key=_requirement_sort_key,
        )
    ]
    if not ordered:
        return {}

    target = section_slug(concern)
    for index, (_key, value) in enumerate(ordered):
        heading = parse_markdown_heading(value)
        if heading is None or section_slug(heading[1]) != target:
            continue
        depth = int(heading[0])
        selected: list[tuple[str, str]] = []

        for prior in range(index - 1, -1, -1):
            parent = parse_markdown_heading(ordered[prior][1])
            if parent is not None and int(parent[0]) < depth:
                selected.append(ordered[prior])
                break

        selected.append(ordered[index])
        for item in ordered[index + 1:]:
            sibling = parse_markdown_heading(item[1])
            if sibling is not None and int(sibling[0]) <= depth:
                break
            selected.append(item)
        return dict(selected)

    return slice_concern_requirements(
        dict(ordered),
        concern=concern,
        require_anchor=True,
    )


def required_atomic_leaf_contract(
    symbol: str,
) -> tuple[str, list[dict[str, Any]]]:
    from .authored_execution_schema import concern_contracts, section_for_symbol

    section = section_for_symbol(symbol)
    concerns = list(concern_contracts(section)) if section else []
    if not section or not concerns:
        raise ValueError(f"IMPLEMENTATION_IR_NONCANONICAL_LEAF: {symbol}")
    return section, concerns


def build_authored_atomic_contract(
    *,
    section: str,
    concerns: Sequence[Mapping[str, Any]],
    requirements: Mapping[str, str],
    raw_obligations: Sequence[str],
    structured_sections: Mapping[str, Any] | None = None,
    production_state_section: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the one canonical contract consumed by atomic production."""

    from .authored_execution_schema import section_spec
    from .authored_structured_design import active_concern_records

    expected = {str(item["concern"]): dict(item) for item in concerns}
    existing: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    extras: list[str] = []
    for raw in raw_obligations:
        decoded = decode_atomic_obligation(raw)
        if decoded is None:
            extras.append(str(raw))
            continue
        payload, instruction = decoded
        name = str(instruction.get("concern") or "").strip()
        if (
            str(instruction.get("section") or "").strip() == section
            and name in expected
        ):
            existing.setdefault(name, (payload, instruction))
            continue
        extras.append(str(raw))

    structured_records = active_concern_records(structured_sections, section)
    section_is_structured = bool(
        isinstance(structured_sections, Mapping)
        and section in structured_sections
    )
    if section == "state_model" and isinstance(production_state_section, Mapping):
        raw_specification = production_state_section.get("specification")
        if isinstance(raw_specification, Mapping):
            structured_records = {
                str(name): [
                    deepcopy(dict(row))
                    for row in rows
                    if isinstance(row, Mapping)
                ]
                for name, rows in raw_specification.items()
                if isinstance(rows, list) and rows
            }
            structured_records.pop("inapplicable_concerns", None)
            section_is_structured = True

    exact_sources: dict[str, dict[str, str]] = {}
    for concern in concerns:
        name = str(concern["concern"])
        if section_is_structured:
            if name not in structured_records:
                continue
            exact_sources[name] = {}
            continue
        source = concern_source_requirements(requirements, concern=name)
        if source:
            exact_sources[name] = source

    if not exact_sources and existing:
        first_name = next(
            (
                str(item["concern"])
                for item in concerns
                if str(item["concern"]) in existing
            ),
            "",
        )
        if first_name:
            payload, _instruction = existing[first_name]
            raw_sources = payload.get("source_requirements")
            if isinstance(raw_sources, Mapping):
                exact_sources[first_name] = {
                    str(key): str(value)
                    for key, value in raw_sources.items()
                    if str(key) in requirements
                }

    spec = section_spec(section) or {}
    drifted: list[str] = []
    active: list[dict[str, Any]] = []
    concern_payloads: dict[str, dict[str, Any]] = {}
    obligations: list[str] = []

    for concern in concerns:
        name = str(concern["concern"])
        if name not in exact_sources:
            continue
        host_sources = exact_sources[name]
        active.append({**dict(concern), "sequence": len(active)})

        if name in existing:
            prior_payload, prior_instruction = deepcopy(existing[name])
            if prior_payload.get("source_requirements") != host_sources:
                drifted.append(name)
            instruction = prior_instruction
        else:
            instruction = {}

        instruction.update({
            "concern": name,
            "concern_template": str(concern["identifier"]),
            "rules": list(concern.get("rules") or []),
            "section": section,
            "section_instruction": str(spec.get("instruction") or ""),
            "task": str(concern["task"]),
        })
        record = {
            "instruction": deepcopy(instruction),
            "source_requirements": deepcopy(host_sources),
            "structured_records": deepcopy(structured_records.get(name, [])),
        }
        concern_payloads[name] = record

        serialized = {
            "instruction": json.dumps(
                instruction,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "source_requirements": deepcopy(host_sources),
        }
        if name in structured_records:
            serialized["structured_records"] = deepcopy(structured_records[name])
        obligations.append(
            json.dumps(serialized, ensure_ascii=False, sort_keys=True)
        )

    return {
        "schema_version": AUTHORED_ATOMIC_CONTRACT_SCHEMA_VERSION,
        "section": section,
        "concerns": concern_payloads,
        "active_concerns": active,
        "implementation_obligations": obligations + extras,
        "drifted_concerns": drifted,
    }


def bind_task_authored_atomic_contract(
    task: dict[str, Any],
    *,
    symbol: str,
    requirements: Mapping[str, str],
    raw_obligations: Sequence[str],
    structured_sections: Mapping[str, Any] | None = None,
    production_state_section: Mapping[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    section, concerns = required_atomic_leaf_contract(symbol)
    contract = build_authored_atomic_contract(
        section=section,
        concerns=concerns,
        requirements=requirements,
        raw_obligations=raw_obligations,
        structured_sections=structured_sections,
        production_state_section=production_state_section,
    )
    task["authored_atomic_contract"] = deepcopy(contract)
    task["implementation_obligations"] = list(
        contract["implementation_obligations"]
    )
    return (
        section,
        deepcopy(contract["active_concerns"]),
        list(contract["drifted_concerns"]),
    )


def task_concern_authority(
    task: Mapping[str, Any],
    concern: str,
) -> dict[str, Any]:
    """Read concern authority from the exact same contract the producer wrote."""

    name = str(concern or "").strip()
    contract = task.get("authored_atomic_contract")
    if isinstance(contract, Mapping):
        schema = str(contract.get("schema_version") or "").strip()
        if schema != AUTHORED_ATOMIC_CONTRACT_SCHEMA_VERSION:
            raise ValueError(
                "AUTHORED_ATOMIC_CONTRACT_SCHEMA_MISMATCH: "
                f"{schema!r}"
            )
        raw_concerns = contract.get("concerns")
        if isinstance(raw_concerns, Mapping):
            raw = raw_concerns.get(name)
            if isinstance(raw, Mapping):
                return {
                    "task_id": str(task.get("task_id") or ""),
                    "section": str(contract.get("section") or ""),
                    "concern": name,
                    "instruction": deepcopy(dict(raw.get("instruction") or {})),
                    "source_requirements": deepcopy(
                        dict(raw.get("source_requirements") or {})
                    ),
                    "structured_records": deepcopy(
                        list(raw.get("structured_records") or [])
                    ),
                }

    raw_obligations = task.get("implementation_obligations")
    if isinstance(raw_obligations, Sequence) and not isinstance(
        raw_obligations, (str, bytes, bytearray)
    ):
        for raw in raw_obligations:
            decoded = decode_atomic_obligation(raw)
            if decoded is None:
                continue
            payload, instruction = decoded
            if str(instruction.get("concern") or "").strip() != name:
                continue
            return {
                "task_id": str(task.get("task_id") or ""),
                "section": str(instruction.get("section") or ""),
                "concern": name,
                "instruction": deepcopy(instruction),
                "source_requirements": deepcopy(
                    dict(payload.get("source_requirements") or {})
                ),
                "structured_records": deepcopy(
                    list(payload.get("structured_records") or [])
                ),
            }

    return {
        "task_id": str(task.get("task_id") or ""),
        "concern": name,
    }


__all__ = [
    "AUTHORED_ATOMIC_CONTRACT_SCHEMA_VERSION",
    "authored_obligation_record",
    "bind_task_authored_atomic_contract",
    "build_authored_atomic_contract",
    "concern_source_requirements",
    "decode_atomic_obligation",
    "merge_authored_obligation_value",
    "required_atomic_leaf_contract",
    "task_concern_authority",
]
