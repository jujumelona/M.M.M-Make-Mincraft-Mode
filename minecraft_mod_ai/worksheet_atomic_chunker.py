from __future__ import annotations

"""Worksheet planner paging with host-owned semantic merge.

The small model never receives a wide concern record. The host deterministically pages
each concern by a bounded field projection, anchors record cardinality on the first page,
and enforces that same row count/order on continuation pages before merging the canonical
worksheet section.
"""

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from typing import Any

from .execution_contract_policy import (
    DEFAULT_ATOMIC_SCHEMA_LIMITS,
    PLANNER_RECORD_ARRAY_ITEM_MAX_CHARS,
    PLANNER_RECORD_FIELD_MAX_CHARS,
    PLANNER_RECORD_PAGE_MAX_FIELDS,
)
from .model_output_atomicity_contract import _assert_closed_object_schemas
from .planning_detail_slots import DETAIL_RECORDS, record_field_schema
from .structured_state_runtime import constrain_state_record_schema
from .planning_detail_template import (
    _PLACEHOLDERS,
    _normalize_section_name,
    _section_description,
    validate_worksheet_section,
)

_CANONICAL_FIELD_DEFAULTS: dict[str, str] = {
    "authority": "server",
    "unit": "count",
    "range": "any",
    "default": "standard",
    "source": "environment",
    "visibility": "public",
    "side_effect": "state update",
    "rejection": "action denied without state change",
    "preserved_state": "all state preserved",
    "cooldown": "immediate (0 ticks)",
    "frequency": "on demand",
    "order": "sequential",
    "non_goal": "out of current requirement scope",
    "limit": "bounded by system memory and tick rate",
    "enforcement": "strict runtime assertion",
    "rollback": "restore prior snapshot",
    "commit": "atomic state commit",
    "time_bound": "under 1 tick (50ms)",
    "memory_bound": "bounded collection",
    "determinism": "deterministic calculation",
    "reentrancy_rule": "thread-safe / non-reentrant",
    "trust_boundary": "client-server boundary validation",
    "dirty_rule": "mark dirty on mutation",
}


class WorksheetConcernChunk(tuple):
    """Tuple-compatible concern selection carrying host-owned field projections."""

    def __new__(
        cls,
        concerns: Sequence[str],
        field_projection: Mapping[str, Sequence[str]],
    ) -> "WorksheetConcernChunk":
        obj = super().__new__(cls, tuple(concerns))
        obj.field_projection = {
            str(concern): tuple(str(field) for field in fields)
            for concern, fields in field_projection.items()
        }
        return obj


def _meaningful_text(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, (list, tuple)):
        return bool(value)
    text = str(value or "").strip()
    return bool(text) and text.casefold() not in _PLACEHOLDERS


def _chunk_projection(
    section: str,
    concerns: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    records = DETAIL_RECORDS[section]
    explicit = getattr(concerns, "field_projection", None)
    projection: dict[str, tuple[str, ...]] = {}
    for concern in concerns:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {section!r}")
        all_fields = tuple(records[concern].split())
        selected = tuple(explicit.get(concern, all_fields)) if isinstance(explicit, Mapping) else all_fields
        if not selected or any(field not in all_fields for field in selected):
            raise ValueError(
                f"Invalid field projection for concern {concern!r} in section {section!r}"
            )
        projection[concern] = selected
    return projection


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    raw = schema.get("type")
    if isinstance(raw, str):
        return {raw}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return {str(item) for item in raw}
    return set()


def _planner_page_field_schema(
    schema: Mapping[str, Any],
    *,
    string_cap: int = PLANNER_RECORD_FIELD_MAX_CHARS,
) -> dict[str, Any]:
    """Clamp one model-facing planner field without changing canonical storage limits."""

    result = deepcopy(dict(schema))
    types = _schema_types(result)
    if "string" in types:
        try:
            explicit = int(result.get("maxLength", string_cap))
        except (TypeError, ValueError):
            explicit = string_cap
        result["maxLength"] = max(1, min(explicit, string_cap))
    if "array" in types:
        try:
            explicit_items = int(
                result.get("maxItems", DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items)
            )
        except (TypeError, ValueError):
            explicit_items = DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items
        result["maxItems"] = max(
            0,
            min(explicit_items, DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items),
        )
        items = result.get("items")
        if isinstance(items, Mapping):
            result["items"] = _planner_page_field_schema(
                items,
                string_cap=min(string_cap, PLANNER_RECORD_ARRAY_ITEM_MAX_CHARS),
            )
    properties = result.get("properties")
    if isinstance(properties, Mapping):
        result["properties"] = {
            str(name): _planner_page_field_schema(
                child,
                string_cap=string_cap,
            )
            if isinstance(child, Mapping)
            else deepcopy(child)
            for name, child in properties.items()
        }
    return result


def pack_section_concerns(
    section: str,
    *,
    max_chunk_size: int | None = None,
) -> list[tuple[str, ...]]:
    """Page every concern deterministically before inference.

    max_chunk_size is retained as a compatibility override for the number of
    record fields exposed per model page. Production uses the central planner page
    width from execution_contract_policy.
    """

    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    if max_chunk_size is not None:
        if max_chunk_size < 1:
            raise ValueError("max_chunk_size must be positive when supplied")
        if max_chunk_size > PLANNER_RECORD_PAGE_MAX_FIELDS:
            raise ValueError(
                "planner page width override cannot exceed the host-owned field bound"
            )
    page_width = max_chunk_size or PLANNER_RECORD_PAGE_MAX_FIELDS

    chunks: list[tuple[str, ...]] = []
    for concern, columns in records.items():
        fields = tuple(columns.split())
        if not fields:
            raise ValueError(
                f"worksheet concern {key}.{concern} has no declared record fields"
            )
        for start in range(0, len(fields), page_width):
            selected_fields = fields[start : start + page_width]
            chunk = WorksheetConcernChunk(
                (concern,),
                {concern: selected_fields},
            )
            schema = worksheet_chunk_schema(
                key,
                chunk,
                include_evidence=False,
                # Static packing validation must exercise the same count-fixed
                # field-page contract as production.
                record_counts={concern: 1},
            )
            _assert_closed_object_schemas(
                schema,
                path=f"worksheet page {key}.{concern}.{start // page_width + 1}",
            )
            chunks.append(chunk)
    return chunks


def _model_transport_schema(schema: Any, *, is_properties_map: bool = False) -> Any:
    if isinstance(schema, dict):
        return {
            key: _model_transport_schema(value, is_properties_map=(key == "properties"))
            for key, value in schema.items()
            if not (key == "pattern" and not is_properties_map)
        }
    if isinstance(schema, list):
        return [_model_transport_schema(value) for value in schema]
    return schema



def worksheet_concern_cardinality_schema(
    section: str,
    concern: str,
) -> dict[str, Any]:
    """Return the tiny semantic decision schema that precedes all field pages."""

    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    if concern not in records:
        raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
    return {
        "type": "object",
        "properties": {
            "record_count": {
                "type": "integer",
                "enum": list(range(DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items + 1)),
            }
        },
        "required": ["record_count"],
        "additionalProperties": False,
    }


def worksheet_concern_cardinality_prompt(
    section: str,
    concern: str,
) -> str:
    """Ask only for semantic record cardinality; the host owns all iteration."""

    key = _normalize_section_name(section)
    if concern not in DETAIL_RECORDS[key]:
        raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
    return "\n".join(
        (
            "ENGINEERING WORKSHEET — bounded concern cardinality decision:",
            f"Section: {key}",
            f"Concern: {concern}",
            f"Purpose: {_section_description(key)}",
            "Choose how many distinct semantic records this concern needs for the user request.",
            f"Return record_count as an integer from 0 through {DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items}.",
            "Use 0 only when this concern is not needed. Do not author record content in this call.",
        )
    )


def worksheet_chunk_schema(
    section: str,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
    record_counts: Mapping[str, int] | None = None,
    model_transport: bool = False,
    state_symbols: Any = None,
) -> dict[str, Any]:
    """Return one bounded planner field-page schema with host-fixed cardinality.

    Cardinality is decided in a separate tiny planner call before any field page.
    Count-free field pages are invalid by construction.
    """
    # Planning owns the canonical worksheet contract. Structured state IR is an
    # authoring/production representation and must not leak back into planning.
    del state_symbols
    key = _normalize_section_name(section)
    active = tuple(concerns)
    if len(active) != 1:
        raise ValueError(
            f"worksheet planner page for {key!r} must contain exactly one concern"
        )
    projection = _chunk_projection(key, concerns)

    properties: dict[str, Any] = {}
    for concern in active:
        fields = projection[concern]
        if len(fields) > PLANNER_RECORD_PAGE_MAX_FIELDS:
            raise ValueError(
                f"worksheet planner page for {key}.{concern} exposes "
                f"{len(fields)} fields; maximum is {PLANNER_RECORD_PAGE_MAX_FIELDS}"
            )
        field_schemas = {
            field: _planner_page_field_schema(
                record_field_schema(key, concern, field)
            )
            for field in fields
        }
        item_schema: dict[str, Any] = {
            "type": "object",
            "properties": field_schemas,
            "required": list(fields),
            "additionalProperties": False,
        }
        if key == "state_model":
            item_schema = constrain_state_record_schema(concern, item_schema)
        if not isinstance(record_counts, Mapping) or concern not in record_counts:
            raise ValueError(
                "worksheet planner field page requires a host-fixed record count for "
                f"{key}.{concern}"
            )
        try:
            count = int(record_counts[concern])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid host record count for {key}.{concern}"
            ) from exc
        if count < 0 or count > DEFAULT_ATOMIC_SCHEMA_LIMITS.max_array_items:
            raise ValueError(
                f"Host record count for {key}.{concern} is outside planner bounds: {count}"
            )
        properties[concern] = {
            "type": "array",
            "minItems": count,
            "maxItems": count,
            "items": item_schema,
        }

    # include_evidence remains a compatibility argument for old callers, but
    # evidence/applicability are host-owned and never widen a planner field page.
    del include_evidence
    required = list(active)
    schema = {
        "type": "object",
        "description": f"Bounded concern field page for {key}: {active[0]}",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if model_transport:
        return _model_transport_schema(schema)
    return schema


def worksheet_chunk_model_schema(
    section: str,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
    record_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return worksheet_chunk_schema(
        section,
        concerns,
        include_evidence=include_evidence,
        record_counts=record_counts,
        model_transport=True,
    )


def validate_worksheet_chunk_signal(
    section: str,
    concerns: Sequence[str],
    chunk: Mapping[str, Any],
) -> dict[str, Any]:
    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    active = tuple(concerns)
    data = dict(chunk)
    nested = data.get("specification")
    if isinstance(nested, Mapping):
        data = dict(nested)

    for concern in active:
        if concern not in records:
            raise ValueError(f"Unknown concern {concern!r} for section {key!r}")
        if concern in data:
            # Presence of an active concern key is sufficient transport signal.
            # The deterministic merge owns shape salvage, semantic cleanup and
            # applicability; malformed scalar values are normalized to no records.
            return dict(chunk)

    raw_inapplicable = data.get("inapplicable_concerns")
    candidates = (
        raw_inapplicable
        if isinstance(raw_inapplicable, list)
        else [raw_inapplicable]
        if isinstance(raw_inapplicable, Mapping)
        else []
    )
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        concern = str(item.get("concern") or "").strip()
        if concern in active and _meaningful_text(item.get("reason")):
            return dict(chunk)

    raise ValueError(
        "DETAILED_PLAN_WORKSHEET_CHUNK: "
        f"{key} concerns {', '.join(active)} contain no meaningful authored content"
    )


def worksheet_chunk_prompt(
    section: str,
    chunk_index: int,
    chunk_count: int,
    concerns: Sequence[str],
    *,
    include_evidence: bool = False,
    record_counts: Mapping[str, int] | None = None,
) -> str:
    from .planning_contract_ssot import schema_skeleton_template

    key = _normalize_section_name(section)
    schema = worksheet_chunk_schema(
        key,
        concerns,
        include_evidence=include_evidence,
        record_counts=record_counts,
    )
    skeleton = schema_skeleton_template(schema)
    projection = _chunk_projection(key, concerns)
    field_text = "; ".join(
        f"{concern}=[{', '.join(fields)}]" for concern, fields in projection.items()
    )
    state_instruction = (
        "For state_model, guard/condition and mutation/initial_state/action are host DSL, "
        "not prose and not Java. Use only the operators and identifiers admitted by the schema. "
        "Put external subsystem actions in algorithm/integration instead of state mutation fields."
        if key == "state_model"
        else ""
    )
    return "\n".join(
        item for item in (
            f"ENGINEERING WORKSHEET — concern chunk {chunk_index}/{chunk_count}:",
            f"Section: {key}",
            f"Active Concerns: {', '.join(concerns)}",
            f"Active Record Fields: {field_text}",
            f"Purpose: {_section_description(key)}",
            "Fill exactly the shown field for the host-fixed record rows.",
            "The host fixed record cardinality in a separate bounded decision. Return exactly "
            + ", ".join(
                f"{name}={count} row(s)" for name, count in record_counts.items()
            )
            + " in the same row order; do not add, remove, or reorder records.",
            "Fill every shown field for every returned row. Keep each value concise and concrete; do not restate the prompt.",
            "Do not invent external facts; the host normalizes harmless omissions only after all field pages are merged.",
            "Return exactly the active concern key and no sibling control metadata.",
            "Never use N/A, none, TODO, TBD, unknown, same-as-above, or another placeholder as the authored content.",
            "DO NOT output JSON Schema keywords (never output 'type', 'properties', 'required', or 'additionalProperties').",
            state_instruction,
            "Return only a JSON object following this data template skeleton:",
            json.dumps(skeleton, ensure_ascii=False, indent=2),
        ) if item
    )


def merge_worksheet_section_chunks(
    section: str,
    chunks: Sequence[Mapping[str, Any]],
    allowed_refs: set[str],
) -> dict[str, Any]:
    from .planning_contract_ssot import is_schema_definition_echo

    key = _normalize_section_name(section)
    records = DETAIL_RECORDS[key]
    expected_chunks = pack_section_concerns(key)
    if len(chunks) != len(expected_chunks):
        raise ValueError(
            f"DETAILED_PLAN_WORKSHEET: {key} expected {len(expected_chunks)} chunks, got {len(chunks)}"
        )

    merged_specification: dict[str, list[dict[str, Any]]] = {
        concern: [] for concern in records
    }
    combined_inapplicable: list[dict[str, Any]] = []
    evidence_refs: list[str] = []

    for chunk, chunk_spec in zip(chunks, expected_chunks, strict=True):
        if not isinstance(chunk, Mapping):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} must be a JSON object"
            )
        if is_schema_definition_echo(chunk):
            raise ValueError(
                f"DETAILED_PLAN_WORKSHEET: chunk for {key} echoed JSON Schema definition instead of data records"
            )
        validate_worksheet_chunk_signal(key, chunk_spec, chunk)
        projection = _chunk_projection(key, chunk_spec)
        chunk_dict = dict(chunk)
        if "specification" in chunk_dict and isinstance(chunk_dict["specification"], Mapping):
            spec = chunk_dict.pop("specification")
            chunk_dict.update(spec)

        for field, value in chunk_dict.items():
            if field == "constraint_evidence_refs":
                if isinstance(value, list):
                    for ref in value:
                        if isinstance(ref, str) and ref.strip() and ref.strip() not in evidence_refs:
                            evidence_refs.append(ref.strip())
                continue
            if field == "inapplicable_concerns":
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, Mapping):
                            concern = str(item.get("concern") or "").strip()
                            reason = str(item.get("reason") or "").strip()
                            if (
                                concern in records
                                and _meaningful_text(reason)
                                and not any(
                                    existing.get("concern") == concern
                                    for existing in combined_inapplicable
                                )
                            ):
                                combined_inapplicable.append(
                                    {"concern": concern, "reason": reason}
                                )
                continue
            if field not in records or field not in projection:
                continue

            if isinstance(value, Mapping):
                candidate_items = [dict(value)]
            elif isinstance(value, list):
                candidate_items = [dict(item) for item in value if isinstance(item, Mapping)]
            else:
                candidate_items = []

            count = len(candidate_items)
            merged_rows = merged_specification[field]
            while len(merged_rows) < count:
                merged_rows.append({})
            selected_fields = projection[field]
            for index, item in enumerate(candidate_items):
                destination = merged_rows[index]
                for field_name in selected_fields:
                    if field_name not in item:
                        continue
                    value_text = item[field_name]
                    if field_name in destination and destination[field_name] != value_text:
                        raise ValueError(
                            "DETAILED_PLAN_WORKSHEET_FIELD_PAGE_CONFLICT: "
                            f"{key}.{field}[{index}].{field_name} changed across pages"
                        )
                    destination[field_name] = value_text

    for concern, columns in records.items():
        expected_fields = tuple(columns.split())
        cleaned_records: list[dict[str, Any]] = []
        for item in merged_specification[concern]:
            if not isinstance(item, Mapping):
                continue
            has_meaningful = any(
                _meaningful_text(item.get(field_name))
                for field_name in expected_fields
            )
            if not has_meaningful:
                continue
            clean_item: dict[str, Any] = {}
            for field_name in expected_fields:
                raw_value = item.get(field_name)
                if (
                    key == "integration"
                    and concern == "initialization_order"
                    and field_name == "prerequisite"
                    and raw_value is None
                ):
                    clean_item[field_name] = "no prerequisite"
                    continue
                field_schema = record_field_schema(key, concern, field_name)
                raw_type = field_schema.get("type")
                array_capable = raw_type == "array" or (
                    isinstance(raw_type, list) and "array" in raw_type
                )
                if isinstance(raw_value, list):
                    clean_item[field_name] = deepcopy(raw_value)
                    continue
                if isinstance(raw_value, Mapping):
                    clean_item[field_name] = deepcopy(raw_value)
                    continue
                if raw_value is None and isinstance(raw_type, list) and "null" in raw_type:
                    clean_item[field_name] = None
                    continue
                val = str(raw_value or "").strip()
                if not val or val.casefold() in _PLACEHOLDERS:
                    if array_capable:
                        clean_item[field_name] = []
                    elif (
                        key == "integration"
                        and concern == "initialization_order"
                        and field_name == "prerequisite"
                    ):
                        clean_item[field_name] = "no prerequisite"
                    else:
                        clean_item[field_name] = _CANONICAL_FIELD_DEFAULTS.get(
                            field_name,
                            f"standard {field_name}",
                        )
                    continue
                clean_item[field_name] = val
            cleaned_records.append(clean_item)
        merged_specification[concern] = cleaned_records

    empty_concerns = {
        concern for concern in records if not merged_specification.get(concern)
    }
    inapplicable_by_concern: dict[str, str] = {}
    for item in combined_inapplicable:
        concern = str(item.get("concern") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if concern in empty_concerns and _meaningful_text(reason):
            inapplicable_by_concern[concern] = reason

    if not any(merged_specification[concern] for concern in records) and not inapplicable_by_concern:
        raise ValueError(
            f"DETAILED_PLAN_WORKSHEET: {key} contains no meaningful authored content"
        )

    final_inapplicable: list[dict[str, str]] = []
    for concern in sorted(empty_concerns):
        reason = inapplicable_by_concern.get(concern)
        if not reason:
            reason = f"No {concern.replace('_', ' ')} required for this {key.replace('_', ' ')}."
        final_inapplicable.append({"concern": concern, "reason": reason})

    valid_evidence_refs = [ref for ref in evidence_refs if ref in allowed_refs]
    assembled_specification: dict[str, Any] = dict(merged_specification)
    assembled_specification["inapplicable_concerns"] = final_inapplicable
    assembled_section = {
        "specification": assembled_specification,
        "constraint_evidence_refs": valid_evidence_refs,
    }
    return validate_worksheet_section(assembled_section, allowed_refs, key)


__all__ = [
    "WorksheetConcernChunk",
    "merge_worksheet_section_chunks",
    "pack_section_concerns",
    "validate_worksheet_chunk_signal",
    "worksheet_chunk_model_schema",
    "worksheet_chunk_prompt",
    "worksheet_chunk_schema",
    "worksheet_concern_cardinality_prompt",
    "worksheet_concern_cardinality_schema",
]
