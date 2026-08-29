from __future__ import annotations

"""Single-pass schema-constrained field generation for planner sections.

Every top-level section field is generated as a small independent unit with its exact
JSON schema embedded in the prompt and supplied to host validation. A field is produced
once. The normal path never patches leaves, replays failed output, feeds validator errors
back to the model, or merges repair payloads.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import agentic_research_game_design as _design
from . import planner_stage_trace as _trace
from .spec import SpecValidationError
from .structured_output import (
    StructuredOutputValidationError,
    validate_structured_output,
)

_CONTROL_KEYS = frozenset({"repair_path", "repair_scope", "repair_tokens"})
_JSONPATH_PREFIXES = ("$.section", "$['section']", '$["section"]')


def _unit_schema(
    field: str,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    schema = properties.get(field)
    if not isinstance(schema, Mapping):
        raise SpecValidationError(
            f"structured section field {field!r} has no JSON schema"
        )
    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": {field: dict(schema)},
                "required": [field],
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }


def _section_schema(
    fields: Sequence[str],
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    missing = [field for field in fields if not isinstance(properties.get(field), Mapping)]
    if missing:
        raise SpecValidationError(
            "structured section fields have no JSON schema: " + ", ".join(missing)
        )
    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": {field: dict(properties[field]) for field in fields},
                "required": list(fields),
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }


def _contains_control_metadata(value: Any) -> bool:
    """Reject host control metadata even when it happens to satisfy the JSON type."""

    if isinstance(value, str):
        return value.strip().startswith(_JSONPATH_PREFIXES)
    if isinstance(value, Mapping):
        if any(str(key) in _CONTROL_KEYS for key in value):
            return True
        return any(
            _contains_control_metadata(key) or _contains_control_metadata(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_control_metadata(item) for item in value)
    return False


def _validated_field(
    raw: str,
    *,
    field: str,
    schema: Mapping[str, Any],
) -> Any:
    validated = validate_structured_output(
        raw,
        response_format="json",
        response_schema=schema,
    )
    try:
        payload = json.loads(validated)
    except json.JSONDecodeError as exc:
        raise SpecValidationError(
            f"{field} constrained output was not JSON: {exc}"
        ) from exc

    section = payload.get("section") if isinstance(payload, Mapping) else None
    if not isinstance(section, Mapping) or field not in section:
        raise SpecValidationError(
            f"{field} constrained output omitted $.section.{field}"
        )
    value = section[field]
    if _contains_control_metadata(value):
        raise SpecValidationError(
            f"{field} emitted host control metadata/JSONPath as semantic content"
        )
    return value


def _schema_bound_messages(
    *,
    prompt: str,
    section_id: str,
    field: str,
    schema: Mapping[str, Any],
    research: Mapping[str, Any],
    accepted: Mapping[str, Any],
) -> list[dict[str, str]]:
    messages = [
        dict(item)
        for item in _design._section_messages(
            prompt=prompt,
            section_id=section_id,
            fields=[field],
            research=research,
            prior_error="",
            prior_candidate=accepted or None,
        )
    ]
    schema_text = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    contract = (
        "This is the only output contract. Return exactly one JSON object matching "
        "the following JSON Schema. Do not emit analysis, markdown, JSONPath, repair "
        "instructions, or fields outside the schema. JSON_SCHEMA="
        + schema_text
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = str(messages[0].get("content") or "") + "\n" + contract
    else:
        messages.insert(0, {"role": "system", "content": contract})
    return messages


def _generate_field_once(
    router: Any,
    *,
    prompt: str,
    section_id: str,
    field: str,
    properties: Mapping[str, Any],
    research: Mapping[str, Any],
    accepted: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace: _trace.PlannerStageTrace,
) -> Any:
    schema = _unit_schema(field, properties)
    raw = ""
    try:
        raw = router.generate_text(
            "planner",
            _schema_bound_messages(
                prompt=prompt,
                section_id=section_id,
                field=field,
                schema=schema,
                research=research,
                accepted=accepted,
            ),
            media_paths=media_paths,
            response_format="json",
            response_schema=schema,
            enable_tools=False,
        )
        value = _validated_field(raw, field=field, schema=schema)
    except StructuredOutputValidationError as exc:
        raw = str(exc.output or raw)
        error = "; ".join(exc.errors)
        trace.record_attempt(
            raw_output=raw,
            validation_error=error,
            candidate=None,
            accepted=None,
            context={
                "section_id": section_id,
                "field": field,
                "generation_strategy": "single_pass_constrained_field",
            },
        )
        raise SpecValidationError(
            f"{section_id}.{field} violated its first-pass schema: {error}"
        ) from exc
    except SpecValidationError as exc:
        trace.record_attempt(
            raw_output=raw,
            validation_error=str(exc),
            candidate=None,
            accepted=None,
            context={
                "section_id": section_id,
                "field": field,
                "generation_strategy": "single_pass_constrained_field",
            },
        )
        raise

    candidate = {field: value}
    trace.record_attempt(
        raw_output=raw,
        validation_error=None,
        candidate=candidate,
        accepted=candidate,
        context={
            "section_id": section_id,
            "field": field,
            "generation_strategy": "single_pass_constrained_field",
        },
    )
    return value


def _generate_section_units(
    router: Any,
    *,
    prompt: str,
    section_id: str,
    fields: Sequence[str],
    properties: Mapping[str, Any],
    research: Mapping[str, Any],
    media_paths: Sequence[str | Path],
    trace_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Generate each schema-bounded field exactly once, then validate the merged section."""

    trace = _trace.PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata={
            "generation_strategy": "single_pass_constrained_field",
            **dict(trace_metadata or {}),
        },
    )
    accepted: dict[str, Any] = {}

    for index, field in enumerate(fields):
        accepted[field] = _generate_field_once(
            router,
            prompt=prompt,
            section_id=section_id,
            field=field,
            properties=properties,
            research=research,
            accepted=accepted,
            media_paths=media_paths if index == 0 else (),
            trace=trace,
        )

    whole_schema = _section_schema(fields, properties)
    whole_raw = json.dumps(
        {"section": accepted},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        validate_structured_output(
            whole_raw,
            response_format="json",
            response_schema=whole_schema,
        )
    except StructuredOutputValidationError as exc:
        raise SpecValidationError(
            f"{section_id} field units passed individually but merged section failed: "
            + "; ".join(exc.errors)
        ) from exc

    trace.record_success(accepted)
    return accepted


__all__ = [
    "_contains_control_metadata",
    "_generate_field_once",
    "_generate_section_units",
    "_schema_bound_messages",
    "_section_schema",
    "_unit_schema",
]
