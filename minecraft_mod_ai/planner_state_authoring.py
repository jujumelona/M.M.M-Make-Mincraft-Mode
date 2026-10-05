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
from .planning_detail_slots import record_field_schema
from .structured_state_runtime import (
    _SUPPORTED_STATE_FUNCTIONS,
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
        for field in requested:
            raw_schema = properties.get(field)
            if not isinstance(raw_schema, Mapping):
                raise ValueError(
                    f"STATE_SEMANTIC_PAGE: missing schema for {concern}.{field}"
                )
            projected_properties[field] = _state_scalar_schema(
                concern, field, raw_schema, fixed
            )

        row_schema = {
            "type": "object",
            "properties": projected_properties,
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

def _state_condition_transport_schema(
    symbols: StateSymbolTable,
) -> dict[str, Any]:
    """Flat, non-recursive model transport for guard/condition authoring."""

    declared = sorted(symbols.declared_names)
    state_name_schema: dict[str, Any]
    if declared:
        state_name_schema = {
            "type": "string",
            "enum": declared,
            "maxLength": 128,
        }
        left_kinds = ["state", "context", "boolean"]
        right_kinds = [
            "none",
            "state",
            "context",
            "number",
            "string",
            "boolean",
            "null",
        ]
    else:
        state_name_schema = {
            "type": "string",
            "const": "",
            "maxLength": 1,
        }
        left_kinds = ["context", "boolean"]
        right_kinds = [
            "none",
            "context",
            "number",
            "string",
            "boolean",
            "null",
        ]

    context_schema = {
        "type": "string",
        "maxLength": 24,
        "pattern": r"^(?:|[A-Za-z_$][A-Za-z0-9_$.]{0,23})$",
    }
    term_schema = {
        "type": "object",
        "properties": {
            "left_kind": {"type": "string", "enum": left_kinds},
            "left_state": state_name_schema,
            "left_context": context_schema,
            "left_value": {
                "type": "string",
                "enum": ["", "true", "false"],
            },
            "operator": {
                "type": "string",
                "enum": ["truthy", "falsey", "==", "!=", ">=", "<=", ">", "<"],
            },
            "right_kind": {"type": "string", "enum": right_kinds},
            "right_state": state_name_schema,
            "right_context": context_schema,
            "right_value": {
                "type": "string",
                "maxLength": 24,
                "pattern": r"^[^{}\[\]]{0,24}$",
            },
        },
        "required": [
            "left_kind",
            "left_state",
            "left_context",
            "left_value",
            "operator",
            "right_kind",
            "right_state",
            "right_context",
            "right_value",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "join": {"type": "string", "enum": ["and", "or"]},
            "terms": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": term_schema,
            },
        },
        "required": ["join", "terms"],
        "additionalProperties": False,
    }


def _condition_operand_from_transport(
    term: Mapping[str, Any],
    *,
    side: str,
) -> dict[str, Any]:
    if side == "left":
        kind = str(term.get("left_kind") or "")
        if kind == "state":
            return {"kind": "state_ref", "name": str(term["left_state"])}
        if kind == "context":
            name = str(term.get("left_context") or "").strip()
            if not name:
                raise ValueError("STATE_CONDITION_TRANSPORT: left context name is required")
            return {"kind": "context_ref", "name": name}
        if kind == "boolean":
            value = str(term.get("left_value") or "").strip().casefold()
            if value not in {"true", "false"}:
                raise ValueError(
                    "STATE_CONDITION_TRANSPORT: boolean left_value must be true or false"
                )
            return {"kind": "literal", "value": value == "true"}
        raise ValueError(f"STATE_CONDITION_TRANSPORT: invalid left kind {kind!r}")

    kind = str(term.get("right_kind") or "")
    if kind == "state":
        return {"kind": "state_ref", "name": str(term["right_state"])}
    if kind == "context":
        name = str(term.get("right_context") or "").strip()
        if not name:
            raise ValueError("STATE_CONDITION_TRANSPORT: right context name is required")
        return {"kind": "context_ref", "name": name}
    if kind == "number":
        value = str(term.get("right_value") or "").strip()
        if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value) is None:
            raise ValueError(
                f"STATE_CONDITION_TRANSPORT: invalid numeric literal {value!r}"
            )
        return {"kind": "number", "value": value}
    if kind == "string":
        return {"kind": "literal", "value": str(term.get("right_value") or "")}
    if kind == "boolean":
        value = str(term.get("right_value") or "").strip().casefold()
        if value not in {"true", "false"}:
            raise ValueError(
                "STATE_CONDITION_TRANSPORT: boolean right_value must be true or false"
            )
        return {"kind": "literal", "value": value == "true"}
    if kind == "null":
        return {"kind": "literal", "value": None}
    raise ValueError(
        f"STATE_CONDITION_TRANSPORT: right operand required for comparison, got {kind!r}"
    )


def _condition_ir_from_transport(
    raw: Mapping[str, Any],
    *,
    symbols: StateSymbolTable,
) -> dict[str, Any]:
    terms = raw.get("terms")
    if not isinstance(terms, Sequence) or isinstance(
        terms, (str, bytes, bytearray)
    ):
        raise ValueError("STATE_CONDITION_TRANSPORT: terms must be an array")

    compiled: list[dict[str, Any]] = []
    for index, item in enumerate(terms):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"STATE_CONDITION_TRANSPORT: terms[{index}] must be an object"
            )
        left = _condition_operand_from_transport(item, side="left")
        operator = str(item.get("operator") or "")
        if operator == "truthy":
            condition = left
        elif operator == "falsey":
            condition = {"kind": "not", "term": left}
        elif operator in {"==", "!=", ">=", "<=", ">", "<"}:
            right = _condition_operand_from_transport(item, side="right")
            condition = {
                "kind": "compare",
                "op": operator,
                "left": left,
                "right": right,
            }
        else:
            raise ValueError(
                f"STATE_CONDITION_TRANSPORT: unsupported operator {operator!r}"
            )
        validate_state_expr_ir(condition, symbols=symbols)
        compiled.append(condition)

    if not compiled:
        raise ValueError("STATE_CONDITION_TRANSPORT: at least one term is required")
    if len(compiled) == 1:
        return compiled[0]

    join = str(raw.get("join") or "")
    if join not in {"and", "or"}:
        raise ValueError(f"STATE_CONDITION_TRANSPORT: invalid join {join!r}")
    result = {"kind": join, "terms": compiled}
    validate_state_expr_ir(result, symbols=symbols)
    return result


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
    schema = _state_condition_transport_schema(symbols)
    instruction = (
        f"Author the boolean condition for state_model.{concern}[{index}].{field}. "
        "Use one or two flat terms only. left_kind=boolean with left_value=true/false "
        "represents an unconditional boolean. operator=truthy/falsey needs no right operand; "
        "for those set right_kind=none and leave right_state/right_context/right_value "
        "empty. For comparisons choose an explicit right_kind and value. "
        "Never emit expression AST keys such as kind, type, left, right, term, or nested "
        "terms; the host builds the canonical typed expression IR."
    )
    raw = generate_fixed_template_value(
        router,
        "planner",
        _state_atomic_messages(
            prompt,
            concern=concern,
            index=index,
            fields=(field,),
            current_row=current,
            symbols_text=symbols.prompt_text(),
            extra_instruction=instruction,
        ),
        response_schema=schema,
        enable_tools=False,
        description=(
            f"Choose flat state condition terms for row {index + 1} of {count} in {concern}."
        ),
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(raw, Mapping):
        raise ValueError("STATE_CONDITION_TRANSPORT: model result must be an object")
    return _condition_ir_from_transport(raw, symbols=symbols)


def _state_mutation_transport_schema(
    symbols: StateSymbolTable,
) -> dict[str, Any]:
    """Flat model transport for state assignments; host owns typed IR assembly."""

    declared = sorted(symbols.declared_names)
    state_name_schema: dict[str, Any]
    value_kinds = [
        "context",
        "number",
        "string",
        "boolean",
        "null",
        "empty_map",
        "empty_list",
    ]
    if declared:
        state_name_schema = {
            "type": "string",
            "enum": declared,
            "maxLength": 128,
        }
        value_kinds.insert(0, "state")
    else:
        state_name_schema = {
            "type": "string",
            "const": "",
            "maxLength": 1,
        }

    context_schema = {
        "type": "string",
        "maxLength": 24,
        "pattern": r"^(?:|[A-Za-z_$][A-Za-z0-9_$.]{0,23})$",
    }
    assignment = {
        "type": "object",
        "properties": {
            "target": state_name_schema,
            "operator": {
                "type": "string",
                "enum": ["=", "+=", "-=", "*=", "/="],
                "maxLength": 2,
            },
            "value_kind": {"type": "string", "enum": value_kinds},
            "value_state": state_name_schema,
            "value_context": context_schema,
            "value_text": {
                "type": "string",
                "maxLength": 24,
                "pattern": r"^[^{}\[\]]{0,24}$",
            },
        },
        "required": [
            "target",
            "operator",
            "value_kind",
            "value_state",
            "value_context",
            "value_text",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": 0,
                "maxItems": 2,
                "items": assignment,
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }


def _mutation_value_from_transport(item: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(item.get("value_kind") or "")
    if kind == "state":
        return {"kind": "state_ref", "name": str(item["value_state"])}
    if kind == "context":
        name = str(item.get("value_context") or "").strip()
        if not name:
            raise ValueError("STATE_MUTATION_TRANSPORT: context name is required")
        return {"kind": "context_ref", "name": name}
    if kind == "number":
        value = str(item.get("value_text") or "").strip()
        if re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", value) is None:
            raise ValueError(
                f"STATE_MUTATION_TRANSPORT: invalid numeric literal {value!r}"
            )
        return {"kind": "number", "value": value}
    if kind == "string":
        return {"kind": "literal", "value": str(item.get("value_text") or "")}
    if kind == "boolean":
        value = str(item.get("value_text") or "").strip().casefold()
        if value not in {"true", "false"}:
            raise ValueError(
                "STATE_MUTATION_TRANSPORT: boolean value_text must be true or false"
            )
        return {"kind": "literal", "value": value == "true"}
    if kind == "null":
        return {"kind": "literal", "value": None}
    if kind == "empty_map":
        return {"kind": "empty_map"}
    if kind == "empty_list":
        return {"kind": "empty_list"}
    raise ValueError(f"STATE_MUTATION_TRANSPORT: invalid value kind {kind!r}")


def _mutation_ir_from_transport(
    raw: Mapping[str, Any],
    *,
    symbols: StateSymbolTable,
) -> list[dict[str, Any]]:
    assignments = raw.get("assignments")
    if not isinstance(assignments, Sequence) or isinstance(
        assignments, (str, bytes, bytearray)
    ):
        raise ValueError("STATE_MUTATION_TRANSPORT: assignments must be an array")

    result: list[dict[str, Any]] = []
    for index, item in enumerate(assignments):
        if not isinstance(item, Mapping):
            raise ValueError(
                f"STATE_MUTATION_TRANSPORT: assignments[{index}] must be an object"
            )
        result.append(
            {
                "target": str(item.get("target") or ""),
                "operator": str(item.get("operator") or ""),
                "value": _mutation_value_from_transport(item),
            }
        )
    validate_mutation_ir(result, symbols=symbols)
    return result


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
    schema = _state_mutation_transport_schema(symbols)
    raw = generate_fixed_template_value(
        router,
        "planner",
        _state_atomic_messages(
            prompt,
            concern=concern,
            index=index,
            fields=(field,),
            current_row=current,
            symbols_text=symbols.prompt_text(),
            extra_instruction=(
                f"Author state assignments for state_model.{concern}[{index}].{field}. "
                "Return only the flat assignments transport. Use an empty assignments "
                "array when this row needs no state mutation. Do not emit nested expression "
                "IR keys such as kind, type, value objects, left, right, terms, or args; "
                "the host constructs and validates canonical typed mutation IR."
            ),
        ),
        response_schema=schema,
        enable_tools=False,
        description=(
            f"Choose flat state assignments for row {index + 1} of {count} in {concern}."
        ),
        output_token_ceiling=structured_output_token_ceiling(schema),
    )
    if not isinstance(raw, Mapping):
        raise ValueError("STATE_MUTATION_TRANSPORT: model result must be an object")
    return _mutation_ir_from_transport(raw, symbols=symbols)


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
    declared_names = sorted(symbols_table.declared_names)
    rows: list[dict[str, Any]] = []
    prior = tuple(existing_rows or ())

    for index in range(count):
        current = (
            dict(prior[index])
            if index < len(prior) and isinstance(prior[index], Mapping)
            else {}
        )
        path = f"state_model.{concern}[{index}].{field}"
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
