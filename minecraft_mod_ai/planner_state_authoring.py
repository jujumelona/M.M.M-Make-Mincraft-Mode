"""Author bounded semantic state choices directly as canonical typed state IR."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import structured_output_token_ceiling
from .structured_state_runtime import (
    _state_variable_value_kind,
    StateSymbolTable,
    state_variable_default_schema,
    validate_mutation_ir,
    validate_state_expr_ir,
)

STATE_EXPRESSION_FIELDS = frozenset({"guard", "condition"})
STATE_MUTATION_FIELDS = frozenset({"mutation", "initial_state", "action"})
STATE_EXECUTABLE_FIELDS = STATE_EXPRESSION_FIELDS | STATE_MUTATION_FIELDS
def _state_atomic_messages(
    prompt: str,
    *,
    concern: str,
    index: int,
    fields: Sequence[str],
    current_row: Mapping[str, Any] | None = None,
    peer_rows: Sequence[Mapping[str, Any]] = (),
    symbols_text: str = "",
    extra_instruction: str = "",
) -> tuple[dict[str, str], ...]:
    """Build an isolated state-authoring turn with no inherited section prompt."""

    requested = ", ".join(str(field) for field in fields)
    user_parts = [
        "Original user request:\n" + str(prompt or "").strip(),
        "Target: state_model." + concern + "[" + str(index) + "]",
        "Author only these fields: " + requested,
    ]
    if current_row:
        user_parts.append(
            "Already-fixed fields for this same row (read-only):\n"
            + json.dumps(dict(current_row), ensure_ascii=True, sort_keys=True, default=str)
        )
    if peer_rows:
        user_parts.append(
            "Already-authored sibling rows (read-only; keep this row distinct where needed):\n"
            + json.dumps(
                [dict(row) for row in peer_rows if isinstance(row, Mapping)],
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
        )
    if symbols_text:
        user_parts.append(symbols_text)
    if extra_instruction:
        user_parts.append(extra_instruction)
    return (
        {
            "role": "system",
            "content": (
                "Fill exactly one host-owned state record projection. Return only the "
                "value or structure explicitly requested for this one field projection. "
                "Never emit a state_model wrapper, a variables/concern array, sibling fields, "
                "Markdown, prose, or loop control. Do not rewrite fields listed as already fixed."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(user_parts),
        },
    )

_STATE_VARIABLE_META_NAMES = frozenset({
    "state",
    "state_model",
    "variable",
    "variables",
    "value",
})


def _used_state_variable_names(
    prior: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    fixed: Mapping[str, Any],
) -> set[str]:
    used = {
        str(candidate.get("name") or "").strip()
        for candidate in (*prior, *rows)
        if isinstance(candidate, Mapping)
        and str(candidate.get("name") or "").strip()
    }
    used.discard(str(fixed.get("name") or "").strip())
    return used


def _semantic_identifier_fragment(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    candidate = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not candidate or not any(char.isalpha() for char in candidate):
        return ""
    if candidate[0].isdigit():
        candidate = "state_" + candidate
    return candidate[:128]


def _canonical_state_variable_name(
    name: str,
    *,
    fixed: Mapping[str, Any],
    used: set[str],
    index: int,
) -> str:
    """Resolve generic/repeated model names to one host-owned stable identifier."""

    candidate = name
    if candidate in _STATE_VARIABLE_META_NAMES or candidate in used:
        for field in ("domain", "unit", "owner"):
            semantic = _semantic_identifier_fragment(fixed.get(field))
            if semantic and semantic not in _STATE_VARIABLE_META_NAMES:
                candidate = semantic
                break
        else:
            candidate = f"state_value_{index + 1}"

    if candidate not in used:
        return candidate

    for ordinal in range(2, 10_000):
        suffix = f"_{ordinal}"
        base = candidate[: max(1, 128 - len(suffix))]
        unique = base + suffix
        if unique not in used:
            return unique
    raise ValueError(
        "STATE_VARIABLE_IDENTIFIER_EXHAUSTED: unable to allocate a unique identifier"
    )


def _state_scalar_schema(
    concern: str,
    field: str,
    schema: Mapping[str, Any],
    current_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Narrow scalar state fields from already-fixed host context when possible."""

    result = deepcopy(dict(schema))
    if concern == "variables" and field == "default":
        family = str(current_row.get("type") or "").strip().casefold()
        return state_variable_default_schema(family)
    return result


def _state_scalar_transport_schema(
    concern: str,
    field: str,
    semantic_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the same finite semantic contract to the structured decoder.

    State defaults are already narrowed from the host-fixed variable type.  Relaxing
    enum/pattern constraints here let malformed values cross the model boundary and
    fail only after inference.  The llama transport projector supports finite enums
    and bounded patterns, so producer and host validation must share one schema.
    """

    del concern, field
    return deepcopy(dict(semantic_schema))


def author_state_semantic_page(
    router: Any,
    prompt: str,
    *,
    concern: str,
    fields: Sequence[str],
    count: int,
    item_schema: Mapping[str, Any],
    existing_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Author one host-fixed state row projection per model call.

    The host owns cardinality, row identity, field names, schema and merging. The
    model chooses only the semantic values for the fields in the current projection.
    """
    if count <= 0:
        return {concern: []}

    requested = tuple(str(field) for field in fields)
    if not requested:
        raise ValueError("STATE_SEMANTIC_PAGE: at least one field is required")
    properties = item_schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("STATE_SEMANTIC_PAGE: item schema has no properties")

    rows: list[dict[str, Any]] = []
    prior = tuple(existing_rows or ())
    for index in range(count):
        fixed = (
            dict(prior[index])
            if index < len(prior) and isinstance(prior[index], Mapping)
            else {}
        )
        projected_properties: dict[str, Any] = {}
        transport_properties: dict[str, Any] = {}
        for field in requested:
            raw_schema = properties.get(field)
            if not isinstance(raw_schema, Mapping):
                raise ValueError(
                    f"STATE_SEMANTIC_PAGE: missing schema for {concern}.{field}"
                )
            semantic_schema = _state_scalar_schema(
                concern, field, raw_schema, fixed
            )
            projected_properties[field] = semantic_schema
            transport_properties[field] = _state_scalar_transport_schema(
                concern, field, semantic_schema
            )

        row_schema = {
            "type": "object",
            "properties": transport_properties,
            "required": list(requested),
            "additionalProperties": False,
        }
        used_names = (
            _used_state_variable_names(prior, rows, fixed)
            if concern == "variables" and "name" in requested
            else set()
        )
        name_instruction = ""
        if concern == "variables" and "name" in requested:
            name_instruction = (
                "For variables.name, choose a concrete identifier for the distinct mutable "
                "concept described by the already-fixed fields for this row. Do not use "
                "container/meta labels such as "
                + ", ".join(sorted(_STATE_VARIABLE_META_NAMES))
                + "."
            )
            if used_names:
                name_instruction += (
                    " Already-used variable names that must not be repeated: "
                    + ", ".join(sorted(used_names))
                    + "."
                )
        messages = _state_atomic_messages(
            prompt,
            concern=concern,
            index=index,
            fields=requested,
            current_row=fixed,
            peer_rows=rows,
            extra_instruction=(
                "Return exactly one JSON object containing only the requested fields for "
                "this row. Do not emit any sibling field, concern array, or wrapper."
                + (" " + name_instruction if name_instruction else "")
            ),
        )
        raw = generate_fixed_template_value(
            router,
            "planner",
            messages,
            response_schema=row_schema,
            enable_tools=False,
            description=(
                f"Author state row projection {concern}[{index}]: "
                + ", ".join(requested)
            ),
            output_token_ceiling=structured_output_token_ceiling(row_schema),
        )
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"STATE_SEMANTIC_PAGE: {concern}[{index}] must be an object"
            )

        row: dict[str, Any] = {}
        for field in requested:
            if field not in raw:
                raise ValueError(
                    f"STATE_SEMANTIC_FIELD_INVALID: {concern}[{index}].{field} "
                    "did not return the required host field"
                )
            value = raw[field]
            field_schema = projected_properties[field]
            errors = tuple(Draft202012Validator(field_schema).iter_errors(value))
            if errors:
                detail = "; ".join(error.message for error in errors[:3])
                raise ValueError(
                    f"STATE_SEMANTIC_FIELD_INVALID: {concern}[{index}].{field}: {detail}"
                )
            if concern == "variables" and field == "name":
                name = str(value).strip()
                value = _canonical_state_variable_name(
                    name,
                    fixed=fixed,
                    used=used_names,
                    index=index,
                )
            row[field] = deepcopy(value)
        rows.append(row)

    return {concern: rows}

def _condition_decision_messages(
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    instruction: str,
) -> tuple[dict[str, str], ...]:
    parts = [
        "Original user request:\n" + str(prompt or "").strip(),
        f"Target condition: state_model.{concern}[{index}].{field}",
    ]
    if current:
        parts.append(
            "Already-fixed fields for this row (read-only):\n"
            + json.dumps(
                dict(current),
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
        )
    symbols_text = symbols.prompt_text()
    if symbols_text:
        parts.append(symbols_text)
    parts.append(instruction)
    return (
        {
            "role": "system",
            "content": (
                "Make exactly the small condition decision requested by the host. "
                "Return one JSON object matching the supplied schema exactly. "
                "Do not emit guard, condition, kind, left, right, terms, state_model, "
                "or any expression-AST wrapper unless that exact key exists in the schema."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(parts),
        },
    )


def _author_condition_object(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    schema: Mapping[str, Any],
    instruction: str,
    description: str,
) -> dict[str, Any]:
    raw = generate_fixed_template_value(
        router,
        "planner",
        _condition_decision_messages(
            prompt,
            concern=concern,
            field=field,
            index=index,
            current=current,
            symbols=symbols,
            instruction=instruction,
        ),
        response_schema=dict(schema),
        enable_tools=False,
        description=description,
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(raw, Mapping):
        raise ValueError(
            "STATE_CONDITION_DECISION: model result must be an object"
        )
    return dict(raw)


def _condition_state_schema(symbols: StateSymbolTable) -> dict[str, Any]:
    declared = sorted(symbols.declared_names)
    if not declared:
        return {"type": "string", "const": "", "maxLength": 1}
    return {
        "type": "string",
        "enum": declared,
        "maxLength": 128,
    }


def _author_condition_left(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    term_index: int,
) -> tuple[dict[str, Any], str]:
    declared = sorted(symbols.declared_names)
    sources = ["context", "true", "false"]
    if declared:
        sources.insert(0, "state")
    schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "enum": sources},
            "state": _condition_state_schema(symbols),
            "context": {
                "type": "string",
                "minLength": 1,
                "maxLength": 24,
                "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]{0,23}$",
            },
            "operator": {
                "type": "string",
                "enum": [
                    "truthy",
                    "falsey",
                    "==",
                    "!=",
                    ">=",
                    "<=",
                    ">",
                    "<",
                ],
            },
        },
        "required": ["source", "state", "context", "operator"],
        "additionalProperties": False,
    }
    raw = _author_condition_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=schema,
        instruction=(
            f"Choose the left operand and operator for condition term {term_index + 1}. "
            "source=state uses state; source=context uses context; source=true/false "
            "is a host boolean literal. The state/context fields are harmless placeholders "
            "when their source is not selected."
        ),
        description=f"Choose condition term {term_index + 1} left operand and operator.",
    )
    source = str(raw["source"])
    operator = str(raw["operator"])
    if source == "state":
        left = {"kind": "state_ref", "name": str(raw["state"])}
    elif source == "context":
        left = {"kind": "context_ref", "name": str(raw["context"])}
    elif source == "true":
        left = {"kind": "literal", "value": True}
    elif source == "false":
        left = {"kind": "literal", "value": False}
    else:
        raise ValueError(
            f"STATE_CONDITION_DECISION: invalid left source {source!r}"
        )
    return left, operator


def _author_condition_right(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    term_index: int,
) -> dict[str, Any]:
    declared = sorted(symbols.declared_names)
    sources = ["context", "number", "string", "true", "false", "null"]
    if declared:
        sources.insert(0, "state")
    kind_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "enum": sources},
        },
        "required": ["source"],
        "additionalProperties": False,
    }
    selected = _author_condition_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=kind_schema,
        instruction=(
            f"Choose only the right operand kind for comparison term {term_index + 1}."
        ),
        description=f"Choose condition term {term_index + 1} right operand kind.",
    )
    source = str(selected["source"])

    if source == "true":
        return {"kind": "literal", "value": True}
    if source == "false":
        return {"kind": "literal", "value": False}
    if source == "null":
        return {"kind": "literal", "value": None}

    if source == "state":
        value_schema = {
            "type": "object",
            "properties": {"value": _condition_state_schema(symbols)},
            "required": ["value"],
            "additionalProperties": False,
        }
    elif source == "context":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 24,
                    "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]{0,23}$",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    elif source == "number":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 24,
                    "pattern": r"^-?[0-9]+(?:\.[0-9]+)?$",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    elif source == "string":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "maxLength": 24,
                    "pattern": r"^[^{}\[\]]{0,24}$",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    else:
        raise ValueError(
            f"STATE_CONDITION_DECISION: invalid right source {source!r}"
        )

    raw_value = _author_condition_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=value_schema,
        instruction=(
            f"Choose only the {source} value for comparison term {term_index + 1}."
        ),
        description=f"Choose condition term {term_index + 1} right operand value.",
    )
    value = raw_value["value"]
    if source == "state":
        return {"kind": "state_ref", "name": str(value)}
    if source == "context":
        return {"kind": "context_ref", "name": str(value)}
    if source == "number":
        return {"kind": "number", "value": str(value)}
    return {"kind": "literal", "value": str(value)}


def _author_state_condition(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    count: int,
    symbols: StateSymbolTable,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    del count
    layout_schema = {
        "type": "object",
        "properties": {
            "layout": {
                "type": "string",
                "enum": ["single", "and", "or"],
            }
        },
        "required": ["layout"],
        "additionalProperties": False,
    }
    layout_raw = _author_condition_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=layout_schema,
        instruction=(
            "Choose condition layout only: single for one predicate, and/or for two predicates."
        ),
        description="Choose condition predicate layout.",
    )
    layout = str(layout_raw["layout"])
    term_count = 1 if layout == "single" else 2

    terms: list[dict[str, Any]] = []
    for term_index in range(term_count):
        left, operator = _author_condition_left(
            router,
            prompt,
            concern=concern,
            field=field,
            index=index,
            current=current,
            symbols=symbols,
            term_index=term_index,
        )
        if operator == "truthy":
            term = left
        elif operator == "falsey":
            term = {"kind": "not", "term": left}
        else:
            right = _author_condition_right(
                router,
                prompt,
                concern=concern,
                field=field,
                index=index,
                current=current,
                symbols=symbols,
                term_index=term_index,
            )
            term = {
                "kind": "compare",
                "op": operator,
                "left": left,
                "right": right,
            }
        validate_state_expr_ir(term, symbols=symbols)
        terms.append(term)

    result = terms[0] if layout == "single" else {
        "kind": layout,
        "terms": terms,
    }
    validate_state_expr_ir(result, symbols=symbols)
    return result


def _mutation_decision_messages(
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    instruction: str,
) -> tuple[dict[str, str], ...]:
    parts = [
        "Original user request:\n" + str(prompt or "").strip(),
        f"Target mutation: state_model.{concern}[{index}].{field}",
    ]
    if current:
        parts.append(
            "Already-fixed fields for this row (read-only):\n"
            + json.dumps(
                dict(current),
                ensure_ascii=True,
                sort_keys=True,
                default=str,
            )
        )
    symbols_text = symbols.prompt_text()
    if symbols_text:
        parts.append(symbols_text)
    parts.append(instruction)
    return (
        {
            "role": "system",
            "content": (
                "Make exactly the small mutation decision requested by the host. "
                "Return one JSON object matching the supplied schema exactly. "
                "Do not emit mutation, action, initial_state, assignments, kind, value, "
                "state_model, or any typed-IR wrapper unless that exact key exists in the schema."
            ),
        },
        {
            "role": "user",
            "content": "\n\n".join(parts),
        },
    )


def _author_mutation_object(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    schema: Mapping[str, Any],
    instruction: str,
    description: str,
) -> dict[str, Any]:
    raw = generate_fixed_template_value(
        router,
        "planner",
        _mutation_decision_messages(
            prompt,
            concern=concern,
            field=field,
            index=index,
            current=current,
            symbols=symbols,
            instruction=instruction,
        ),
        response_schema=dict(schema),
        enable_tools=False,
        description=description,
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(raw, Mapping):
        raise ValueError(
            "STATE_MUTATION_DECISION: model result must be an object"
        )
    return dict(raw)


def _mutation_target_family(
    symbols: StateSymbolTable,
    target: str,
) -> str:
    return _state_variable_value_kind(symbols.variables.get(target))


def _mutation_compatible_state_names(
    symbols: StateSymbolTable,
    target_family: str,
) -> list[str]:
    result: list[str] = []
    for name in sorted(symbols.declared_names):
        family = _state_variable_value_kind(symbols.variables.get(name))
        if target_family == "unknown" or family in {target_family, "unknown"}:
            result.append(name)
    return result


def _mutation_operator_choices(target_family: str) -> list[str]:
    if target_family == "number":
        return ["=", "+=", "-=", "*=", "/="]
    if target_family == "string":
        return ["=", "+="]
    if target_family in {"boolean", "map", "list"}:
        return ["="]
    return ["=", "+=", "-=", "*=", "/="]


def _mutation_value_source_choices(
    target_family: str,
    *,
    has_compatible_state: bool,
) -> list[str]:
    if target_family == "number":
        values = ["context", "number", "null"]
    elif target_family == "boolean":
        values = ["context", "true", "false", "null"]
    elif target_family == "string":
        values = ["context", "string", "null"]
    else:
        values = [
            "context",
            "number",
            "string",
            "true",
            "false",
            "null",
        ]
    if has_compatible_state:
        values.insert(0, "state")
    return values


def _author_mutation_value(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    current: Mapping[str, Any],
    symbols: StateSymbolTable,
    assignment_index: int,
    target: str,
) -> dict[str, Any]:
    target_family = _mutation_target_family(symbols, target)
    compatible_states = _mutation_compatible_state_names(
        symbols,
        target_family,
    )
    sources = _mutation_value_source_choices(
        target_family,
        has_compatible_state=bool(compatible_states),
    )
    source_schema = {
        "type": "object",
        "properties": {
            "source": {"type": "string", "enum": sources},
        },
        "required": ["source"],
        "additionalProperties": False,
    }
    raw_source = _author_mutation_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=source_schema,
        instruction=(
            f"Choose only the value source for assignment {assignment_index + 1} "
            f"to {target!r}. The target value family is {target_family!r}."
        ),
        description=(
            f"Choose mutation assignment {assignment_index + 1} value source."
        ),
    )
    source = str(raw_source["source"])

    if source == "true":
        return {"kind": "literal", "value": True}
    if source == "false":
        return {"kind": "literal", "value": False}
    if source == "null":
        return {"kind": "literal", "value": None}

    if source == "state":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "enum": compatible_states,
                    "maxLength": 128,
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    elif source == "context":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 24,
                    "pattern": r"^[A-Za-z_$][A-Za-z0-9_$.]{0,23}$",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    elif source == "number":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 24,
                    "pattern": r"^-?[0-9]+(?:\.[0-9]+)?$",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    elif source == "string":
        value_schema = {
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                    "maxLength": 24,
                    "pattern": r"^[^{}\[\]]{0,24}$",
                }
            },
            "required": ["value"],
            "additionalProperties": False,
        }
    else:
        raise ValueError(
            f"STATE_MUTATION_DECISION: invalid value source {source!r}"
        )

    raw_value = _author_mutation_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=value_schema,
        instruction=(
            f"Choose only the {source} value for assignment {assignment_index + 1} "
            f"to {target!r}."
        ),
        description=(
            f"Choose mutation assignment {assignment_index + 1} scalar value."
        ),
    )
    value = raw_value["value"]
    if source == "state":
        return {"kind": "state_ref", "name": str(value)}
    if source == "context":
        return {"kind": "context_ref", "name": str(value)}
    if source == "number":
        return {"kind": "number", "value": str(value)}
    return {"kind": "literal", "value": str(value)}


def _author_state_mutation(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    index: int,
    count: int,
    symbols: StateSymbolTable,
    current: Mapping[str, Any],
) -> list[dict[str, Any]]:
    del count
    declared = sorted(symbols.declared_names)
    if not declared:
        return []

    count_schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "string",
                "enum": ["zero", "one", "two"],
            }
        },
        "required": ["count"],
        "additionalProperties": False,
    }
    raw_count = _author_mutation_object(
        router,
        prompt,
        concern=concern,
        field=field,
        index=index,
        current=current,
        symbols=symbols,
        schema=count_schema,
        instruction=(
            "Choose only how many state assignments this row requires: zero, one, or two."
        ),
        description="Choose state mutation assignment count.",
    )
    count_name = str(raw_count["count"])
    assignment_count = {"zero": 0, "one": 1, "two": 2}[count_name]

    result: list[dict[str, Any]] = []
    for assignment_index in range(assignment_count):
        target_schema = {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": declared,
                    "maxLength": 128,
                }
            },
            "required": ["target"],
            "additionalProperties": False,
        }
        raw_target = _author_mutation_object(
            router,
            prompt,
            concern=concern,
            field=field,
            index=index,
            current=current,
            symbols=symbols,
            schema=target_schema,
            instruction=(
                f"Choose only the target state variable for assignment {assignment_index + 1}."
            ),
            description=(
                f"Choose mutation assignment {assignment_index + 1} target."
            ),
        )
        target = str(raw_target["target"])
        family = _mutation_target_family(symbols, target)

        operator_schema = {
            "type": "object",
            "properties": {
                "operator": {
                    "type": "string",
                    "enum": _mutation_operator_choices(family),
                }
            },
            "required": ["operator"],
            "additionalProperties": False,
        }
        raw_operator = _author_mutation_object(
            router,
            prompt,
            concern=concern,
            field=field,
            index=index,
            current=current,
            symbols=symbols,
            schema=operator_schema,
            instruction=(
                f"Choose only the operator for assignment {assignment_index + 1} "
                f"to {target!r}, whose value family is {family!r}."
            ),
            description=(
                f"Choose mutation assignment {assignment_index + 1} operator."
            ),
        )
        value = _author_mutation_value(
            router,
            prompt,
            concern=concern,
            field=field,
            index=index,
            current=current,
            symbols=symbols,
            assignment_index=assignment_index,
            target=target,
        )
        result.append(
            {
                "target": target,
                "operator": str(raw_operator["operator"]),
                "value": value,
            }
        )

    validate_mutation_ir(result, symbols=symbols)
    return result


def author_state_field_page(
    router: Any,
    prompt: str,
    *,
    concern: str,
    field: str,
    count: int,
    symbols: Any,
    existing_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return canonical typed-IR rows after finite, validated semantic decisions."""
    if count <= 0:
        return {concern: []}

    symbols_table = (
        symbols
        if isinstance(symbols, StateSymbolTable)
        else StateSymbolTable(symbols or ())
    )
    rows: list[dict[str, Any]] = []
    prior = tuple(existing_rows or ())

    for index in range(count):
        current = (
            dict(prior[index])
            if index < len(prior) and isinstance(prior[index], Mapping)
            else {}
        )
        if field in STATE_EXPRESSION_FIELDS:
            value = _author_state_condition(
                router,
                prompt,
                concern=concern,
                field=field,
                index=index,
                count=count,
                symbols=symbols_table,
                current=current,
            )
        else:
            value = _author_state_mutation(
                router,
                prompt,
                concern=concern,
                field=field,
                index=index,
                count=count,
                symbols=symbols_table,
                current=current,
            )

        rows.append({field: value})

    return {concern: rows}
