from __future__ import annotations

"""Decompose central-intelligence structured outputs into physically atomic model calls.

Central specialists and reviewers may reason in parallel, but each individual model
response must stay within the small-model atomicity envelope.  This adapter preserves
the public merged payload while splitting wide nested objects into bounded fragments
that the host deterministically joins.
"""

import json
from functools import wraps
from types import ModuleType
from typing import Any, Mapping, Sequence

_MARKER = "_mmm_central_atomic_fixed_template_generation"
_MAX_ITEMS = 4
_MAX_CHARS = 256


def _bounded_property(schema: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(schema)
    kind = value.get("type")
    if kind == "string":
        value.setdefault("maxLength", _MAX_CHARS)
    elif kind == "array":
        value.setdefault("maxItems", _MAX_ITEMS)
        items = value.get("items")
        if isinstance(items, Mapping):
            item = dict(items)
            if item.get("type") == "string":
                item.setdefault("maxLength", _MAX_CHARS)
            value["items"] = item
    return value


def _fragment_schema(
    field: str,
    source_schema: Mapping[str, Any],
    names: Sequence[str],
) -> dict[str, Any]:
    outer = source_schema["properties"][field]
    properties = outer["properties"]
    fragment_properties = {
        name: _bounded_property(properties[name])
        for name in names
    }
    return {
        "type": "object",
        "properties": {
            field: {
                "type": "object",
                "properties": fragment_properties,
                "required": list(names),
                "additionalProperties": False,
            }
        },
        "required": [field],
        "additionalProperties": False,
    }


def _groups_for(central_module: ModuleType, response_schema: Mapping[str, Any]):
    if response_schema is central_module._COUNCIL_SCHEMA:
        return "analysis", (
            ("must_preserve", "must_not_invent", "subproblems"),
            ("risks", "research_questions", "confidence"),
        )
    if response_schema is central_module._CHAIR_SCHEMA:
        return "synthesis", (
            ("requirements", "negative_constraints", "subproblem_order"),
            ("acceptance_observables", "unresolved_questions"),
        )
    if response_schema is central_module._REVIEW_SCHEMA:
        return "review", (
            ("missing_requirements", "unsupported_additions", "contradictions"),
            ("research_gaps", "affected_sections", "severity"),
            ("confidence",),
        )
    return None


def install(central_module: ModuleType) -> None:
    current = central_module.generate_fixed_template_text
    if bool(getattr(current, _MARKER, False)):
        return

    @wraps(current)
    def atomic_generate(
        router: Any,
        role: str,
        messages: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ) -> str:
        response_schema = kwargs.get("response_schema")
        if not isinstance(response_schema, Mapping):
            return current(router, role, messages, **kwargs)
        plan = _groups_for(central_module, response_schema)
        if plan is None:
            return current(router, role, messages, **kwargs)

        field, groups = plan
        merged: dict[str, Any] = {}
        for names in groups:
            fragment_kwargs = dict(kwargs)
            fragment_kwargs["response_schema"] = _fragment_schema(
                field,
                response_schema,
                names,
            )
            raw = current(router, role, messages, **fragment_kwargs)
            value = json.loads(raw)
            fragment = value.get(field)
            if not isinstance(fragment, Mapping):
                raise ValueError(
                    f"central atomic fragment omitted {field!r} object"
                )
            merged.update(fragment)
        return json.dumps({field: merged}, ensure_ascii=False, separators=(",", ":"))

    setattr(atomic_generate, _MARKER, True)
    atomic_generate.__wrapped__ = current
    central_module.generate_fixed_template_text = atomic_generate


__all__ = ["install"]
