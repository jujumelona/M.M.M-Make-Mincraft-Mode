"""Author bounded semantic state choices directly as canonical typed state IR."""
from __future__ import annotations

import json
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
    mutations_schema,
    state_expr_schema,
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

def _unique_state_identifier(
    value: str,
    *,
    prior_rows: Sequence[Mapping[str, Any]],
    row_index: int,
) -> str:
    """Make an already-valid internal state identifier unique deterministically."""

    base = str(value or "").strip()
    used = {
        str(row.get("name") or "").strip()
        for row in prior_rows
        if isinstance(row, Mapping)
    }
    if base not in used:
        return base

    suffix = max(2, int(row_index) + 1)
    while True:
        suffix_text = f"_{suffix}"
        candidate = base[: max(1, 128 - len(suffix_text))] + suffix_text
        if candidate not in used:
            return candidate
        suffix += 1

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
                value = _unique_state_identifier(
                    str(value),
                    prior_rows=rows,
                    row_index=index,
                )
            row[field] = deepcopy(value)
        rows.append(row)

    return {concern: rows}

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
            inner_schema = state_expr_schema(symbols_table)
            row_schema = {
                "type": "object",
                "properties": {field: inner_schema},
                "required": [field],
                "additionalProperties": False,
            }
            instruction = (
                f"Author state expression for {path}. "
                f"Return complete semantic IR object. "
                f"Declared state variables: {', '.join(declared_names) or 'none'}."
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
                    symbols_text=symbols_table.prompt_text(),
                    extra_instruction=instruction,
                ),
                response_schema=row_schema,
                enable_tools=False,
                description=f"Author state {field} for row {index + 1} of {count} in {concern}.",
                output_token_ceiling=structured_output_token_ceiling(row_schema),
            )
            raw_value = raw.get(field, raw) if isinstance(raw, Mapping) else raw
            validate_state_expr_ir(raw_value, symbols=symbols_table)
            value = deepcopy(raw_value)
        else:
            inner_schema = mutations_schema(symbols_table)
            row_schema = {
                "type": "object",
                "properties": {field: inner_schema},
                "required": [field],
                "additionalProperties": False,
            }
            typed_symbols = symbols_table.prompt_text()
            instruction = (
                f"Author state mutation for {path}. "
                f"Return list of state assignments or empty list for no state mutation. "
                "Each assignment value must use the host IR branch compatible with its "
                "declared target type. Never serialize a JSON object/array into a string "
                "literal; compound state must use an explicitly supported container IR. "
                f"Declared state variables: {', '.join(declared_names) or 'none'}."
                + (f"\n{typed_symbols}" if typed_symbols else "")
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
                    symbols_text=typed_symbols,
                    extra_instruction=instruction,
                ),
                response_schema=row_schema,
                enable_tools=False,
                description=f"Author state {field} for row {index + 1} of {count} in {concern}.",
                output_token_ceiling=structured_output_token_ceiling(row_schema),
            )
            raw_value = raw.get(field, raw) if isinstance(raw, Mapping) else raw
            validate_mutation_ir(raw_value, symbols=symbols_table)
            value = deepcopy(raw_value)

        rows.append({field: value})

    return {concern: rows}
