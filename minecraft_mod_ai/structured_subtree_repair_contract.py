from __future__ import annotations

"""Exact JSON-path repair layered over the structured planner contract.

The first section generation stays permissive so a mostly-good section is never thrown
away just because one leaf has the wrong shape. Once validation locates a bad leaf, the
repair turn is constrained by the schema for exactly that leaf. Valid siblings remain
host-owned frozen state. Repair continues only while the validator frontier changes;
there is no arbitrary retry count.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import agentic_research_game_design as _design
from . import planner_stage_trace as _trace
from . import structured_repair_contract as _base
from .spec import SpecValidationError

_MISSING = object()


def _path(tokens: Sequence[str | int]) -> str:
    result = "$.section"
    for token in tokens:
        result = _base._json_path(result, token)
    return result


def _schema_at(
    properties: Mapping[str, Any], tokens: Sequence[str | int]
) -> Mapping[str, Any] | None:
    if not tokens or not isinstance(tokens[0], str):
        return None
    schema = properties.get(tokens[0])
    if not isinstance(schema, Mapping):
        return None
    for token in tokens[1:]:
        expected = schema.get("type")
        if isinstance(token, int):
            if expected != "array":
                return None
            child = schema.get("items")
        else:
            if expected != "object":
                return None
            props = schema.get("properties")
            child = props.get(token) if isinstance(props, Mapping) else None
            if child is None:
                additional = schema.get("additionalProperties")
                child = additional if isinstance(additional, Mapping) else None
        if not isinstance(child, Mapping):
            return None
        schema = child
    return schema


def _value_at(root: Any, tokens: Sequence[str | int]) -> Any:
    value = root
    for token in tokens:
        if isinstance(token, int):
            if not isinstance(value, list) or not 0 <= token < len(value):
                return _MISSING
            value = value[token]
        else:
            if not isinstance(value, Mapping) or token not in value:
                return _MISSING
            value = value[token]
    return value


def _set_at(root: dict[str, Any], tokens: Sequence[str | int], value: Any) -> None:
    if not tokens:
        raise SpecValidationError("structured repair cannot replace the section root")
    parent: Any = root
    for token in tokens[:-1]:
        if isinstance(token, int):
            if not isinstance(parent, list) or not 0 <= token < len(parent):
                raise SpecValidationError(
                    f"structured repair path disappeared: {_path(tokens)}"
                )
            parent = parent[token]
        else:
            if not isinstance(parent, Mapping) or token not in parent:
                raise SpecValidationError(
                    f"structured repair path disappeared: {_path(tokens)}"
                )
            parent = parent[token]
    leaf = tokens[-1]
    if isinstance(leaf, int):
        if not isinstance(parent, list) or not 0 <= leaf < len(parent):
            raise SpecValidationError(
                f"structured repair list path disappeared: {_path(tokens)}"
            )
        parent[leaf] = value
    else:
        if not isinstance(parent, dict):
            raise SpecValidationError(
                f"structured repair object path disappeared: {_path(tokens)}"
            )
        parent[leaf] = value


def _delete_at(root: dict[str, Any], tokens: Sequence[str | int]) -> bool:
    if not tokens:
        return False
    parent: Any = root
    for token in tokens[:-1]:
        parent = _value_at(parent, (token,))
        if parent is _MISSING:
            return False
    leaf = tokens[-1]
    if isinstance(leaf, str) and isinstance(parent, dict) and leaf in parent:
        del parent[leaf]
        return True
    return False


def _diag(
    code: str,
    tokens: Sequence[str | int],
    *,
    message: str,
    expected: Any = None,
    observed: Any = None,
    repair_tokens: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    target = tuple(tokens if repair_tokens is None else repair_tokens)
    return {
        "code": code,
        "path": _path(tokens),
        "message": message,
        "expected": expected,
        "observed_type": _base._observed_type(observed),
        "repair_scope": _path(target),
        "repair_tokens": list(target),
    }


def _validate(
    value: Any,
    schema: Mapping[str, Any],
    tokens: tuple[str | int, ...],
) -> list[dict[str, Any]]:
    expected = schema.get("type")
    if expected == "string":
        if not isinstance(value, str):
            return [
                _diag(
                    "invalid_type",
                    tokens,
                    message="expected string",
                    expected="string",
                    observed=value,
                )
            ]
        minimum = int(schema.get("minLength", 0) or 0)
        if minimum and len(value) < minimum:
            return [
                _diag(
                    "min_length",
                    tokens,
                    message="string is shorter than schema minimum",
                    expected={"minLength": minimum},
                    observed=value,
                )
            ]
        maximum = schema.get("maxLength")
        if maximum is not None and len(value) > int(maximum):
            return [
                _diag(
                    "max_length",
                    tokens,
                    message="string exceeds schema maximum",
                    expected={"maxLength": maximum},
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
                    tokens,
                    message="expected boolean",
                    expected="boolean",
                    observed=value,
                )
            ]
        )
    if expected == "number":
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        return (
            []
            if valid
            else [
                _diag(
                    "invalid_type",
                    tokens,
                    message="expected number",
                    expected="number",
                    observed=value,
                )
            ]
        )
    if expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
        return (
            []
            if valid
            else [
                _diag(
                    "invalid_type",
                    tokens,
                    message="expected integer",
                    expected="integer",
                    observed=value,
                )
            ]
        )

    if expected == "array":
        if not isinstance(value, list):
            return [
                _diag(
                    "invalid_type",
                    tokens,
                    message="expected array",
                    expected="array",
                    observed=value,
                )
            ]
        diagnostics: list[dict[str, Any]] = []
        minimum = schema.get("minItems")
        if minimum is not None and len(value) < int(minimum):
            diagnostics.append(
                _diag(
                    "min_items",
                    tokens,
                    message="array is shorter than schema minimum",
                    expected={"minItems": minimum},
                    observed=value,
                )
            )
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > int(maximum):
            diagnostics.append(
                _diag(
                    "max_items",
                    tokens,
                    message="array exceeds schema maximum",
                    expected={"maxItems": maximum},
                    observed=value,
                )
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                diagnostics.extend(_validate(item, item_schema, (*tokens, index)))
        return diagnostics

    if expected == "object":
        if not isinstance(value, Mapping):
            return [
                _diag(
                    "invalid_type",
                    tokens,
                    message="expected object",
                    expected="object",
                    observed=value,
                )
            ]
        diagnostics = []
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for raw_key in required:
                key = str(raw_key)
                if key not in value:
                    diagnostics.append(
                        _diag(
                            "missing_required",
                            (*tokens, key),
                            message="required field is missing",
                            expected="present",
                            observed=None,
                        )
                    )
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, Mapping):
                diagnostics.extend(
                    _validate(value[key], child_schema, (*tokens, str(key)))
                )
        additional = schema.get("additionalProperties", True)
        known = set(properties)
        if additional is False:
            for key in value:
                if key not in known:
                    diagnostics.append(
                        _diag(
                            "unexpected_field",
                            (*tokens, str(key)),
                            message="field is not allowed by schema",
                            expected="absent",
                            observed=value[key],
                        )
                    )
        elif isinstance(additional, Mapping):
            for key, child in value.items():
                if key not in known:
                    diagnostics.extend(
                        _validate(child, additional, (*tokens, str(key)))
                    )
        return diagnostics

    return []


def _diagnostics(
    section: Mapping[str, Any] | None,
    *,
    fields: Sequence[str],
    properties: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if section is None:
        return [
            _diag(
                "invalid_json_object",
                (),
                message="planner response did not expose a usable section object",
                expected="object",
                observed=None,
                repair_tokens=(),
            )
        ]
    result: list[dict[str, Any]] = []
    for field in fields:
        tokens = (field,)
        if field not in section:
            result.append(
                _diag(
                    "missing_required",
                    tokens,
                    message="required section field is missing",
                    expected="present",
                    observed=None,
                )
            )
            continue
        schema = properties.get(field)
        if isinstance(schema, Mapping):
            result.extend(_validate(section[field], schema, tokens))
    return result


def _target_for(diagnostic: Mapping[str, Any]) -> tuple[str | int, ...]:
    raw = diagnostic.get("repair_tokens")
    if not isinstance(raw, list):
        return ()
    tokens: list[str | int] = []
    for token in raw:
        if isinstance(token, bool):
            return ()
        if isinstance(token, (str, int)):
            tokens.append(token)
        else:
            return ()
    return tuple(tokens)


def _repair_schema(target_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Constrain only the replacement leaf, never the already-valid section."""

    return {
        "type": "object",
        "properties": {"repair": dict(target_schema)},
        "required": ["repair"],
        "additionalProperties": False,
    }


def _repair_messages(
    *,
    prompt: str,
    section_id: str,
    target: Sequence[str | int],
    diagnostic: Mapping[str, Any],
    current_value: Any,
    working: Mapping[str, Any],
    research: Mapping[str, Any],
) -> list[dict[str, str]]:
    system = (
        "You are an exact JSON-path repair worker. Return one JSON object with exactly "
        "one key named repair. The value of repair replaces only repair_path. Every other "
        "JSON path is frozen host state and must not be repeated or reinterpreted. Correct "
        "the validator diagnostic while preserving the authoritative request semantics."
    )
    parent_tokens = tuple(target[:-1])
    parent_value = _value_at(working, parent_tokens) if parent_tokens else working
    payload = {
        "authoritative_request": prompt,
        "section_id": section_id,
        "repair_path": _path(target),
        "validator_diagnostic": dict(diagnostic),
        "current_value": None if current_value is _MISSING else current_value,
        "frozen_parent_context": parent_value,
        "research": _design._compact_research_for_design(research),
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ),
        },
    ]


def _repair_value(raw: str) -> Any:
    try:
        payload = _design._extract_json_object(raw)
    except SpecValidationError:
        return _MISSING
    return payload["repair"] if "repair" in payload else _MISSING


def _diagnostic_key(diagnostic: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(diagnostic.get("path") or ""),
        str(diagnostic.get("code") or ""),
    )


def _generate_section_exact(
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
            "repair_strategy": "json_path_exact",
            **dict(trace_metadata or {}),
        },
    )

    # Keep the first turn permissive. Its valid siblings are valuable host state even
    # when one field has the wrong JSON type.
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
    candidate = _base._candidate_from_raw(raw, fields)
    diagnostics = _diagnostics(candidate, fields=fields, properties=properties)
    trace.record_attempt(
        raw_output=raw,
        validation_error=(
            None
            if not diagnostics
            else "; ".join(str(item["message"]) for item in diagnostics)
        ),
        candidate=candidate,
        accepted=candidate if not diagnostics else None,
        context={
            "section_id": section_id,
            "repair_strategy": "json_path_exact",
        },
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

    if candidate is None:
        return _base._generate_section_local(
            router,
            prompt=prompt,
            section_id=section_id,
            fields=fields,
            properties=properties,
            research=research,
            media_paths=media_paths,
            trace_metadata={
                "fallback_from": "json_path_exact",
                **dict(trace_metadata or {}),
            },
        )

    working = dict(candidate)
    while diagnostics:
        diagnostic = diagnostics[0]
        target = _target_for(diagnostic)

        if diagnostic.get("code") == "unexpected_field" and target:
            if not _delete_at(working, target):
                raise SpecValidationError(
                    f"failed to remove unexpected field at {_path(target)}"
                )
            diagnostics = _diagnostics(
                working,
                fields=fields,
                properties=properties,
            )
            trace.record_attempt(
                raw_output="<host-delete-unexpected-field>",
                validation_error=(
                    None
                    if not diagnostics
                    else "; ".join(str(item["message"]) for item in diagnostics)
                ),
                candidate=working,
                accepted=working if not diagnostics else None,
                context={
                    "section_id": section_id,
                    "repair_strategy": "json_path_exact",
                    "host_action": "delete_unexpected_field",
                },
                diagnostics=diagnostics,
                repair_scope=[_path(target)],
            )
            continue

        target_schema = _schema_at(properties, target)
        if not target or not isinstance(target_schema, Mapping):
            return _base._generate_section_local(
                router,
                prompt=prompt,
                section_id=section_id,
                fields=fields,
                properties=properties,
                research=research,
                media_paths=media_paths,
                trace_metadata={
                    "fallback_from": "json_path_exact",
                    **dict(trace_metadata or {}),
                },
            )

        key = _diagnostic_key(diagnostic)
        current = _value_at(working, target)
        repair_raw = router.generate_text(
            "planner",
            _repair_messages(
                prompt=prompt,
                section_id=section_id,
                target=target,
                diagnostic=diagnostic,
                current_value=current,
                working=working,
                research=research,
            ),
            media_paths=media_paths,
            response_format="json",
            response_schema=_repair_schema(target_schema),
            enable_tools=False,
        )
        value = _repair_value(repair_raw)
        if value is _MISSING:
            raise SpecValidationError(
                f"{section_id} exact-path repair returned no replacement at {_path(target)}"
            )
        _set_at(working, target, value)

        next_diagnostics = _diagnostics(
            working,
            fields=fields,
            properties=properties,
        )
        next_keys = {_diagnostic_key(item) for item in next_diagnostics}
        if key in next_keys:
            raise SpecValidationError(
                f"{section_id} exact-path repair made no validator progress at "
                f"{key[0]} ({key[1]})"
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
                "repair_strategy": "json_path_exact",
                "repair_path": _path(target),
                "validator_frontier_changed": True,
            },
            diagnostics=diagnostics,
            repair_scope=[_path(target)],
        )

    result = {field: working[field] for field in fields}
    trace.record_success(result)
    return result


__all__ = [
    "_diagnostics",
    "_generate_section_exact",
    "_repair_schema",
    "_schema_at",
]
