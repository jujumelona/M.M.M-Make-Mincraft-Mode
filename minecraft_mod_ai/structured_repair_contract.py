from __future__ import annotations

"""Field-local structured-output repair and typed planner diagnostics.

A valid part of a structured planner response is evidence: do not discard it because a
sibling field failed validation. This contract freezes valid fields, requests only the
invalid/missing subtree, merges the patch host-side, and records machine-readable JSON-path
diagnostics. Repair continues only while the validator frontier changes.
"""

import json
import sys
from collections.abc import Mapping, Sequence
from functools import wraps
from pathlib import Path
from typing import Any

from . import agentic_research_game_design as _design
from . import planner_stage_trace as _trace
from .spec import SpecValidationError

_INSTALLED = False
_TRACE_SCHEMA_V2 = "mmm/planner-stage-trace-v2"


def _json_path(parent: str, child: str | int) -> str:
    if isinstance(child, int):
        return f"{parent}[{child}]"
    if child.isidentifier():
        return f"{parent}.{child}"
    return f"{parent}[{json.dumps(child, ensure_ascii=False)}]"


def _observed_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def _diag(
    code: str,
    path: str,
    *,
    message: str,
    expected: Any = None,
    observed: Any = None,
    repair_scope: str | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "path": path,
        "message": message,
        "expected": expected,
        "observed_type": _observed_type(observed),
        "repair_scope": repair_scope or path,
    }


def _validate_value(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
) -> list[dict[str, Any]]:
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            return [
                _diag(
                    "invalid_type",
                    path,
                    message="expected string",
                    expected="string",
                    observed=value,
                )
            ]
        if int(schema.get("minLength", 0) or 0) and len(value) < int(schema["minLength"]):
            return [
                _diag(
                    "min_length",
                    path,
                    message="string is shorter than schema minimum",
                    expected={"minLength": schema["minLength"]},
                    observed=value,
                )
            ]
        if schema.get("maxLength") is not None and len(value) > int(schema["maxLength"]):
            return [
                _diag(
                    "max_length",
                    path,
                    message="string exceeds schema maximum",
                    expected={"maxLength": schema["maxLength"]},
                    observed=value,
                )
            ]
        return []
    if expected == "boolean":
        return (
            []
            if isinstance(value, bool)
            else [
                _diag(
                    "invalid_type",
                    path,
                    message="expected boolean",
                    expected="boolean",
                    observed=value,
                )
            ]
        )
    if expected == "number":
        return (
            []
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else [
                _diag(
                    "invalid_type",
                    path,
                    message="expected number",
                    expected="number",
                    observed=value,
                )
            ]
        )
    if expected == "array":
        if not isinstance(value, list):
            return [
                _diag(
                    "invalid_type",
                    path,
                    message="expected array",
                    expected="array",
                    observed=value,
                )
            ]
        diagnostics: list[dict[str, Any]] = []
        if schema.get("maxItems") is not None and len(value) > int(schema["maxItems"]):
            diagnostics.append(
                _diag(
                    "max_items",
                    path,
                    message="array exceeds schema maximum",
                    expected={"maxItems": schema["maxItems"]},
                    observed=value,
                )
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                diagnostics.extend(
                    _validate_value(item, item_schema, _json_path(path, index))
                )
        return diagnostics
    if expected == "object":
        if not isinstance(value, Mapping):
            return [
                _diag(
                    "invalid_type",
                    path,
                    message="expected object",
                    expected="object",
                    observed=value,
                )
            ]
        diagnostics: list[dict[str, Any]] = []
        properties = schema.get("properties")
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                key_str = str(key)
                if key_str not in value:
                    diagnostics.append(
                        _diag(
                            "missing_required",
                            _json_path(path, key_str),
                            message="required field is missing",
                            expected="present",
                            observed=None,
                            repair_scope=path,
                        )
                    )
        if isinstance(properties, Mapping):
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, Mapping):
                    diagnostics.extend(
                        _validate_value(
                            value[key], child_schema, _json_path(path, str(key))
                        )
                    )
        additional = schema.get("additionalProperties", True)
        known = set(properties) if isinstance(properties, Mapping) else set()
        if additional is False:
            for key in value:
                if key not in known:
                    diagnostics.append(
                        _diag(
                            "unexpected_field",
                            _json_path(path, str(key)),
                            message="field is not allowed by schema",
                            expected="absent",
                            observed=value[key],
                            repair_scope=path,
                        )
                    )
        elif isinstance(additional, Mapping):
            for key, child in value.items():
                if key not in known:
                    diagnostics.extend(
                        _validate_value(child, additional, _json_path(path, str(key)))
                    )
        return diagnostics
    return []


def _section_diagnostics(
    section: Mapping[str, Any] | None,
    *,
    fields: Sequence[str],
    properties: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if section is None:
        return [
            _diag(
                "invalid_json_object",
                "$.section",
                message="planner response did not expose a usable section object",
                expected="object",
                observed=None,
                repair_scope="$.section",
            )
        ]
    diagnostics: list[dict[str, Any]] = []
    for field in fields:
        path = _json_path("$.section", field)
        if field not in section:
            diagnostics.append(
                _diag(
                    "missing_required",
                    path,
                    message="required section field is missing",
                    expected="present",
                    observed=None,
                    repair_scope=path,
                )
            )
            continue
        schema = properties.get(field)
        if isinstance(schema, Mapping):
            for diagnostic in _validate_value(section[field], schema, path):
                diagnostic["repair_scope"] = path
                diagnostics.append(diagnostic)
    return diagnostics


def _repair_fields(
    diagnostics: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[str]:
    selected: list[str] = []
    for field in fields:
        prefix = _json_path("$.section", field)
        if any(
            str(item.get("repair_scope") or item.get("path") or "").startswith(prefix)
            for item in diagnostics
        ):
            selected.append(field)
    return selected or list(fields)


def _repair_schema(
    repair_fields: Sequence[str],
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    subset = {
        field: properties[field] for field in repair_fields if field in properties
    }
    return {
        "type": "object",
        "properties": {
            "repair": {
                "type": "object",
                "properties": subset,
                "required": list(repair_fields),
                "additionalProperties": False,
            }
        },
        "required": ["repair"],
        "additionalProperties": False,
    }


def _repair_messages(
    *,
    prompt: str,
    section_id: str,
    repair_fields: Sequence[str],
    frozen_section: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    research: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are a JSON subtree repair worker. The host has already accepted and frozen "
        "all fields not listed in repair_fields. Return only one JSON object named repair "
        "containing exactly repair_fields. Never repeat, rewrite, summarize, or reinterpret "
        "frozen_section. Correct only the validator diagnostics. Preserve the authoritative "
        "request semantics and do not invent filler merely to satisfy a type."
    )
    payload = {
        "authoritative_request": prompt,
        "section_id": section_id,
        "repair_fields": list(repair_fields),
        "validator_diagnostics": [dict(item) for item in diagnostics],
        "frozen_section": dict(frozen_section),
        "research": _design._compact_research_for_design(research),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, sort_keys=True),
        },
    ]


def _candidate_from_raw(
    raw: str,
    fields: Sequence[str],
) -> dict[str, Any] | None:
    try:
        payload = _design._extract_json_object(raw)
    except SpecValidationError:
        return None
    section = payload.get("section")
    if isinstance(section, Mapping):
        return {str(key): value for key, value in section.items()}
    direct = {field: payload[field] for field in fields if field in payload}
    return direct or None


def _repair_from_raw(
    raw: str,
    repair_fields: Sequence[str],
) -> dict[str, Any] | None:
    try:
        payload = _design._extract_json_object(raw)
    except SpecValidationError:
        return None
    repair = payload.get("repair")
    if isinstance(repair, Mapping):
        return {field: repair[field] for field in repair_fields if field in repair}
    direct = {field: payload[field] for field in repair_fields if field in payload}
    return direct or None


def _diagnostic_frontier(
    diagnostics: Sequence[Mapping[str, Any]],
) -> frozenset[tuple[str, str]]:
    return frozenset(
        (
            str(item.get("path") or ""),
            str(item.get("code") or ""),
        )
        for item in diagnostics
    )


def _generate_section_local(
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
    trace = _trace.PlannerStageTrace(
        stage=f"game_design_{section_id}",
        prompt=prompt,
        media_paths=media_paths,
        metadata={
            "repair_strategy": "field_local",
            **dict(trace_metadata or {}),
        },
    )
    raw = router.generate_text(
        "planner",
        _design._section_messages(
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            research=research,
            prior_error="",
            prior_candidate=None,
        ),
        media_paths=media_paths,
        response_format="text",
        response_schema=None,
        enable_tools=False,
    )
    candidate = _candidate_from_raw(raw, fields)
    diagnostics = _section_diagnostics(
        candidate,
        fields=fields,
        properties=properties,
    )
    trace.record_attempt(
        raw_output=raw,
        validation_error=(
            None
            if not diagnostics
            else "; ".join(str(item["message"]) for item in diagnostics)
        ),
        candidate=candidate,
        accepted=candidate if not diagnostics else None,
        context={"section_id": section_id, "repair_strategy": "field_local"},
        diagnostics=diagnostics,
        repair_scope=[
            str(item.get("repair_scope") or item.get("path"))
            for item in diagnostics
        ],
    )
    if not diagnostics and candidate is not None:
        result = {field: candidate[field] for field in fields}
        trace.record_success(result)
        return result

    working = dict(candidate or {})
    while diagnostics:
        repair_fields = _repair_fields(diagnostics, fields)
        frozen = {
            field: working[field]
            for field in fields
            if field in working and field not in repair_fields
        }
        frontier = _diagnostic_frontier(diagnostics)
        repair_raw = router.generate_text(
            "planner",
            _repair_messages(
                prompt=prompt,
                section_id=section_id,
                repair_fields=repair_fields,
                frozen_section=frozen,
                diagnostics=diagnostics,
                research=research,
            ),
            media_paths=media_paths,
            response_format="json",
            response_schema=_repair_schema(repair_fields, properties),
            enable_tools=False,
        )
        patch = _repair_from_raw(repair_raw, repair_fields)
        if not patch:
            paths = ", ".join(
                str(item.get("path") or "$") for item in diagnostics
            )
            raise SpecValidationError(
                f"{section_id} field-local repair returned no usable patch at {paths}"
            )
        for field in repair_fields:
            if field in patch:
                working[field] = patch[field]

        next_diagnostics = _section_diagnostics(
            working,
            fields=fields,
            properties=properties,
        )
        if frontier == _diagnostic_frontier(next_diagnostics):
            paths = ", ".join(
                str(item.get("path") or "$") for item in next_diagnostics
            )
            raise SpecValidationError(
                f"{section_id} field-local repair made no validator progress at {paths}"
            )
        diagnostics = next_diagnostics
        trace.record_attempt(
            raw_output=repair_raw,
            validation_error=(
                None
                if not diagnostics
                else "; ".join(str(item["message"]) for item in diagnostics)
            ),
            candidate=working,
            accepted=working if not diagnostics else None,
            context={
                "section_id": section_id,
                "repair_strategy": "field_local",
                "frozen_fields": sorted(frozen),
                "repair_fields": repair_fields,
                "validator_frontier_changed": True,
            },
            diagnostics=diagnostics,
            repair_scope=[
                _json_path("$.section", field) for field in repair_fields
            ],
        )

    result = {field: working[field] for field in fields}
    trace.record_success(result)
    return result


def _install_trace_v2() -> None:
    cls = _trace.PlannerStageTrace
    original = cls.record_attempt
    if getattr(original, "_mmm_typed_diagnostics", False):
        return

    @wraps(original)
    def record_attempt(
        self: Any,
        *,
        raw_output: str,
        validation_error: str | None,
        candidate: Mapping[str, Any] | None = None,
        accepted: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        diagnostics: Sequence[Mapping[str, Any]] | None = None,
        repair_scope: Sequence[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        with _trace._TRACE_LOCK:
            index = self._attempt_index
            self._attempt_index += 1
            payload = {
                "schema_version": _TRACE_SCHEMA_V2,
                "run_id": self.run_id,
                "stage": self.stage,
                "attempt_index": index,
                "prompt_sha256": self.prompt_sha256,
                "raw_output": raw_output,
                "raw_output_sha256": _trace._sha256_text(raw_output),
                "validation_error": validation_error,
                "diagnostics": _trace._json_safe(
                    [dict(item) for item in diagnostics or ()]
                ),
                "repair_scope": list(repair_scope or ()),
                "candidate": (
                    _trace._json_safe(dict(candidate))
                    if candidate is not None
                    else None
                ),
                "accepted": (
                    _trace._json_safe(dict(accepted))
                    if accepted is not None
                    else None
                ),
                "context": dict(context or {}),
            }
            self._write_json(
                self.directory / f"attempt-{index:06d}.json",
                payload,
            )
            with (self.directory / "attempts.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    + "\n"
                )

    record_attempt._mmm_typed_diagnostics = True
    cls.record_attempt = record_attempt
    _trace._TRACE_SCHEMA = _TRACE_SCHEMA_V2


def install_structured_repair_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _install_trace_v2()
    original = _design._generate_section
    if not getattr(original, "_mmm_field_local_repair", False):

        @wraps(original)
        def generate_section(
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
            return _generate_section_local(
                router,
                prompt=prompt,
                section_id=section_id,
                fields=fields,
                properties=properties,
                research=research,
                media_paths=media_paths,
                trace_metadata=trace_metadata,
            )

        generate_section._mmm_field_local_repair = True
        _design._generate_section = generate_section

        for name, module in tuple(sys.modules.items()):
            if not name.startswith("minecraft_mod_ai.") or module is None:
                continue
            if getattr(module, "_generate_section", None) is original:
                setattr(module, "_generate_section", generate_section)
    _INSTALLED = True


__all__ = ["install_structured_repair_contract"]
