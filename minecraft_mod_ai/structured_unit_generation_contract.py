from __future__ import annotations

"""Constrained field-unit generation for structured planner sections.

Each top-level section field is generated under its own JSON schema. A rejected field is
regenerated as a whole unit; no leaf patching, type coercion, or JSONPath-as-data recovery
is performed.
"""

import json
import os
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


def _attempt_limit() -> int:
    raw = os.environ.get("MMM_STRUCTURED_UNIT_ATTEMPTS", "").strip()
    try:
        value = int(raw) if raw else 3
    except ValueError:
        value = 3
    return max(1, min(value, 5))


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
    return {
        "type": "object",
        "properties": {
            "section": {
                "type": "object",
                "properties": {
                    field: dict(properties[field])
                    for field in fields
                    if isinstance(properties.get(field), Mapping)
                },
                "required": list(fields),
                "additionalProperties": False,
            }
        },
        "required": ["section"],
        "additionalProperties": False,
    }


def _contains_control_metadata(value: Any) -> bool:
    """Reject host control paths/keys even when they happen to satisfy JSON types."""

    if isinstance(value, str):
        stripped = value.strip()
        return stripped.startswith(_JSONPATH_PREFIXES)
    if isinstance(value, Mapping):
        if any(str(key) in _CONTROL_KEYS for key in value):
            return True
        return any(
            _contains_control_metadata(key) or _contains_control_metadata(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
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


def _validation_error(exc: BaseException) -> str:
    if isinstance(exc, StructuredOutputValidationError):
        return "; ".join(exc.errors)
    return f"{type(exc).__name__}: {exc}"


def _raw_from_error(exc: BaseException) -> str:
    if isinstance(exc, StructuredOutputValidationError):
        return str(exc.output or "")
    return ""


def _generate_field(
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
    prior_error = ""
    last_error = ""

    for attempt in range(1, _attempt_limit() + 1):
        raw = ""
        try:
            raw = router.generate_text(
                "planner",
                _design._section_messages(
                    prompt=prompt,
                    section_id=section_id,
                    fields=[field],
                    research=research,
                    prior_error=prior_error,
                    prior_candidate=accepted or None,
                ),
                media_paths=media_paths,
                response_format="json",
                response_schema=schema,
                enable_tools=False,
            )
            value = _validated_field(raw, field=field, schema=schema)
        except (StructuredOutputValidationError, SpecValidationError) as exc:
            if not raw:
                raw = _raw_from_error(exc)
            last_error = _validation_error(exc)
            trace.record_attempt(
                raw_output=raw,
                validation_error=last_error,
                candidate=None,
                accepted=None,
                context={
                    "section_id": section_id,
                    "field": field,
                    "generation_strategy": "constrained_field_unit",
                    "attempt": attempt,
                    "action": "regenerate_entire_field",
                },
            )
            prior_error = (
                f"The previous complete value for field {field!r} was rejected by the "
                f"host validator. Regenerate that entire field from the authoritative "
                f"request; do not patch a leaf and do not output JSONPath/control metadata. "
                f"Validator: {last_error}"
            )
            continue

        candidate = {field: value}
        trace.record_attempt(
            raw_output=raw,
            validation_error=None,
            candidate=candidate,
            accepted=candidate,
            context={
                "section_id": section_id,
                "field": field,
                "generation_strategy": "constrained_field_unit",
                "attempt": attempt,
            },
        )
        return value

    raise SpecValidationError(
        f"{section_id}.{field} did not satisfy its constrained schema after "
        f"{_attempt_limit()} complete-field generations: {last_error}"
    )


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
    """Generate a section one schema-constrained top-level field at a time."""

    trace = _trace.PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata={
            "generation_strategy": "constrained_field_unit",
            **dict(trace_metadata or {}),
        },
    )
    accepted: dict[str, Any] = {}

    for index, field in enumerate(fields):
        accepted[field] = _generate_field(
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
    "_attempt_limit",
    "_contains_control_metadata",
    "_generate_section_units",
    "_section_schema",
    "_unit_schema",
]
