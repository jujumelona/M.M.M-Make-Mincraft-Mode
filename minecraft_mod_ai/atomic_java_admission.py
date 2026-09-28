"""Admit native Java components independently, preserving accepted model work.

The required native tool transport stays unchanged. A renderer/schema failure is
owned here, where its structured response still exists, rather than being turned
into an empty source string by the outer concern retry loop.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .custom_module_errors import AtomicJavaDecisionError, CustomModuleGenerationError
from .model_adapters.base import NativeToolDecisionRejected


def rejected_decision(exc: NativeToolDecisionRejected, tool_name: str) -> Any:
    """Salvage only one identified, schema-rejected native call, never prose."""
    if exc.tool_name != tool_name or len(exc.rejections) != 1:
        return None
    rejection = exc.rejections[0]
    if rejection.get("failure_code") != "TOOL_SCHEMA_INVALID":
        return None
    if rejection.get("original_tool") != tool_name:
        return None
    try:
        value = json.loads(str(rejection.get("raw_arguments") or ""))
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _declaration(value: Any) -> Any:
    """Keep symbols/types visible without repeating accepted method bodies."""
    if isinstance(value, Mapping):
        return {key: _declaration(item) for key, item in value.items()
                if key not in {"body", "initializer"}}
    if isinstance(value, list):
        return [_declaration(item) for item in value]
    return value


def _references_name(value: Any, name: str, rewrite: Callable[[Any, Mapping[str, str]], str]) -> bool:
    if isinstance(value, Mapping):
        return any(_references_name(item, name, rewrite) for item in value.values())
    if isinstance(value, list):
        return any(_references_name(item, name, rewrite) for item in value)
    return isinstance(value, str) and rewrite(value, {name: "$mmm$renamed"}) != value



_NON_CONCRETE_COLLECTION_TYPES = frozenset({
    "Collection", "List", "Set", "Map", "Queue", "Deque",
    "SortedSet", "NavigableSet", "SortedMap", "NavigableMap", "EnumSet",
})


def _semantic_component_issue(
    category: str,
    component: Mapping[str, Any],
) -> tuple[str, list[str]]:
    if category != "fields":
        return "", []
    initializer = str(component.get("initializer") or "").strip()
    if not initializer:
        return "", []

    direct = re.search(
        r"\bnew\s+(?:java\.util\.)?([A-Za-z_$][A-Za-z0-9_$]*)"
        r"\s*(?:<[^;(){}]*>)?\s*\(",
        initializer,
    )
    if direct and direct.group(1) in _NON_CONCRETE_COLLECTION_TYPES:
        return (
            f"{direct.group(1)} cannot be instantiated directly; use a concrete "
            "implementation or a valid factory expression.",
            ["initializer"],
        )

    if re.search(
        r"\b(?:java\.util\.)?EnumSet\s*\.\s*noneOf\s*\(\s*"
        r"(?:java\.lang\.)?Enum\s*\.\s*class\s*\)",
        initializer,
    ):
        return (
            "EnumSet.noneOf requires a concrete enum class; java.lang.Enum.class "
            "does not identify an element enum.",
            ["initializer"],
        )
    return "", []


def admit_components(
    decision: Any,
    *,
    parameters: Mapping[str, Any],
    render: Callable[[Mapping[str, Any]], str],
    repair: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
    attempt_limit: int,
    canonical_name: Callable[[str], str],
    rewrite_identifiers: Callable[[Any, Mapping[str, str]], str],
    reserved_names: tuple[str, ...] = (),
) -> str:
    """Repair only rejected members; the model cannot replace accepted siblings.

    There are finitely many slots in the received response. Each slot has a
    bounded correction budget and an actual-response repeat detector. Terminal
    component failures do not re-enter whole-region generation.
    """
    if not isinstance(decision, Mapping):
        raise AtomicJavaDecisionError(
            "ATOMIC_CONCERN_RESPONSE_INVALID: emit_java_structure returned a non-object.",
            response=decision,
        )
    working = deepcopy(dict(decision))
    properties = parameters["properties"]
    unknown = sorted(set(working) - set(properties))
    if unknown:
        raise AtomicJavaDecisionError(
            f"ATOMIC_CONCERN_RESPONSE_INVALID: member structure has unexpected fields: {unknown}",
            response=working,
        )
    # Root/category shape cannot be assigned to a single member. Keep the whole
    # native evidence for the outer bounded retry, without pretending it is Java.
    for category, values in working.items():
        if not isinstance(values, list):
            raise AtomicJavaDecisionError(
                f"ATOMIC_CONCERN_RESPONSE_INVALID: {category} must be an array.",
                response=working,
            )

    type_names = {canonical_name(name) for name in reserved_names if name}
    accepted: list[dict[str, Any]] = []
    for category, values in working.items():
        schema = properties[category]["items"]
        for index, initial in enumerate(values):
            component = initial
            seen: set[str] = set()
            for attempt in range(attempt_limit + 1):
                errors = list(Draft202012Validator(schema).iter_errors(component))
                reason = errors[0].message if errors else ""
                path = list(errors[0].absolute_path) if errors else []
                if not reason and category in {"classes", "records", "enums"}:
                    name = canonical_name(component.get("name", ""))
                    if name in type_names:
                        reason = f"Nested type {name!r} collides with an existing or host-owned type."
                        path = ["name"]
                if not reason:
                    reason, path = _semantic_component_issue(category, component)
                if not reason:
                    try:
                        render({category: [component]})
                    except CustomModuleGenerationError as exc:
                        reason = str(exc)
                if not reason:
                    values[index] = component
                    if category in {"classes", "records", "enums"}:
                        type_names.add(canonical_name(component["name"]))
                    accepted.append({"category": category, "index": index,
                                     "declaration": _declaration(component)})
                    break

                evidence = json.dumps(component, ensure_ascii=False, sort_keys=True)
                if evidence in seen or attempt == attempt_limit:
                    raise AtomicJavaDecisionError(
                        "ATOMIC_COMPONENT_REPAIR_NO_PROGRESS: "
                        f"{category}[{index}] remained invalid: {reason}",
                        response={**working, category: [*values[:index], component, *values[index + 1:]]},
                    )
                seen.add(evidence)
                if (path == ["name"] and category in {"classes", "records", "enums"}
                        and isinstance(component.get("name"), str)):
                    # A declaration-only rename would silently rebind recursive
                    # fields/returns to the host outer type. Let the model repair
                    # this whole component when its internal bindings depend on
                    # the rejected name; all sibling components remain frozen.
                    body = {key: value for key, value in component.items() if key != "name"}
                    if _references_name(body, component["name"], rewrite_identifiers):
                        path = []
                selected_schema = schema
                rejected_value = component
                for key in path:
                    selected_schema = (selected_schema["items"] if isinstance(key, int)
                                       else selected_schema["properties"][key])
                    rejected_value = rejected_value[key]
                repair_schema = {
                    "type": "object",
                    "properties": {"value": deepcopy(selected_schema)},
                    "required": ["value"],
                    "additionalProperties": False,
                }
                if category in {"classes", "records", "enums"}:
                    if path == ["name"]:
                        repair_schema["properties"]["value"]["not"] = {"enum": sorted(type_names)}
                    elif not path:
                        repair_schema["properties"]["value"]["properties"]["name"]["not"] = {
                            "enum": sorted(type_names),
                        }
                context = {
                    "category": category,
                    "index": index,
                    "selected_path": path,
                    "rejected_value": rejected_value,
                    "component_declaration": _declaration(component),
                    "validation_error": reason,
                    "reserved_type_names": sorted(type_names),
                    "accepted_components": accepted,
                    "other_component_declarations": [
                        {"category": other_category, "index": other_index,
                         "declaration": _declaration(other)}
                        for other_category, others in working.items()
                        for other_index, other in enumerate(others)
                        if (other_category, other_index) != (category, index)
                    ],
                    "rule": (
                        "Return only the replacement value at selected_path. An empty path selects "
                        "the whole component. Preserve its runtime responsibility. "
                        "Accepted components and the host outer class are immutable. "
                        "A nested helper must have its own name, never the outer class name."
                    ),
                }
                from .root_cause_trace import emit_root_cause

                emit_root_cause(
                    "atomic_java_component_rejected", stage="production",
                    operation="repair_java_component", gate="component_admission",
                    result="RETRY", reason=reason,
                    details={"category": category, "index": index, "attempt": attempt + 1,
                             "response": working, "rejected_component": component},
                )
                try:
                    replacement = repair(repair_schema, context)
                except NativeToolDecisionRejected as exc:
                    replacement = rejected_decision(exc, "repair_java_component")
                    if replacement is None or Draft202012Validator(repair_schema).is_valid(replacement):
                        raise AtomicJavaDecisionError(
                            f"ATOMIC_COMPONENT_REPAIR_REJECTED: {exc}", response=list(exc.rejections),
                        ) from exc
                if not isinstance(replacement, Mapping) or set(replacement) != {"value"}:
                    raise AtomicJavaDecisionError(
                        "ATOMIC_COMPONENT_REPAIR_REJECTED: expected exactly one replacement value.",
                        response=replacement,
                    )
                if not path:
                    component = replacement["value"]
                else:
                    updated = deepcopy(component)
                    parent = updated
                    for key in path[:-1]:
                        parent = parent[key]
                    parent[path[-1]] = replacement["value"]
                    component = updated
    try:
        errors = list(Draft202012Validator(parameters).iter_errors(working))
        if errors:
            raise CustomModuleGenerationError(errors[0].message)
        return render(working)
    except CustomModuleGenerationError as exc:
        raise AtomicJavaDecisionError(
            f"ATOMIC_COMPONENT_ASSEMBLY_REJECTED: {exc}", response=working,
        ) from exc
