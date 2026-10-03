"""Single Source of Truth (SSOT) contract for state_model.

Unifies model schema, final validator, and compiler under a single canonical
structured IR representation. No other module should independently define state
record field schemas.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from .structured_state_runtime import (
    StateSymbolTable,
    compile_mutation_ir as compile_state_mutations,
    compile_state_expr_ir as compile_state_expression,
    constrain_state_record_schema,
    mutations_schema as _runtime_mutations_schema,
    render_state_model_concern,
    state_concern_schema as _runtime_state_concern_schema,
    state_expr_schema as _runtime_state_expr_schema,
    validate_mutation_ir,
    validate_state_concern,
    validate_state_expr_ir,
    validate_structured_state_section as validate_state_section,
)


def expression_schema(
    symbols: Any = None,
    *,
    allow_string: bool = False,
) -> dict[str, Any]:
    """Canonical schema for state expression fields (guard, condition)."""
    ir_schema = _runtime_state_expr_schema(symbols)
    if allow_string:
        return {
            "anyOf": [
                ir_schema,
                {"type": "string", "maxLength": 512},
            ],
            "description": "Host state-compiler DSL expression or structured IR.",
        }
    return ir_schema


def mutation_schema(
    symbols: Any = None,
    *,
    allow_string: bool = False,
) -> dict[str, Any]:
    """Canonical schema for state mutation fields (mutations, mutation, initial_state, action)."""
    ir_schema = _runtime_mutations_schema(symbols)
    if allow_string:
        return {
            "anyOf": [
                ir_schema,
                {"type": "string", "maxLength": 512},
            ],
            "description": "Host state-compiler DSL mutation statements or structured IR array.",
        }
    return ir_schema


def variable_schema() -> dict[str, Any]:
    """Canonical schema for state_model variables concern record."""
    return state_concern_schema("variables")


def transition_schema(
    symbols: Any = None,
    *,
    allow_string: bool = False,
) -> dict[str, Any]:
    """Canonical schema for state_model transitions concern record."""
    return state_concern_schema("transitions", allowed_state_symbols=symbols, allow_string=allow_string)


def state_concern_schema(
    concern: str,
    *,
    allowed_state_symbols: Any = None,
    allow_string: bool = False,
) -> dict[str, Any]:
    """Canonical schema for a single state_model concern record.

    When allow_string is True (e.g. host validation for legacy test plans), state
    DSL strings are accepted alongside the structured IR. When allow_string is False
    (model transport), only the canonical structured IR is accepted.
    """
    schema = deepcopy(_runtime_state_concern_schema(concern, allowed_state_symbols=allowed_state_symbols))
    if allow_string:
        for field in ("guard", "condition"):
            if field in schema.get("properties", {}):
                desc = schema["properties"][field].get("description")
                schema["properties"][field] = expression_schema(allowed_state_symbols, allow_string=True)
                if desc:
                    schema["properties"][field]["description"] = desc
        for field in ("mutations", "mutation", "initial_state", "action"):
            if field in schema.get("properties", {}):
                desc = schema["properties"][field].get("description")
                schema["properties"][field] = mutation_schema(allowed_state_symbols, allow_string=True)
                if desc:
                    schema["properties"][field]["description"] = desc
        if concern == "transitions":
            schema["required"] = ["from_state", "trigger", "guard", "to_state"]
    return schema


def state_section_schema(
    allowed_state_symbols: Any = None,
    *,
    allow_string: bool = True,
) -> dict[str, Any]:
    """Canonical schema for the entire state_model specification section.

    Used by final validator and specification_schema so host validation never
    drifts from the model or compiler contracts.
    """
    from .task_template_catalog import detail_records

    records = detail_records()["state_model"]
    properties: dict[str, Any] = {
        concern: {
            "type": "array",
            "maxItems": 4,
            "items": state_concern_schema(
                concern,
                allowed_state_symbols=allowed_state_symbols,
                allow_string=allow_string,
            ),
        }
        for concern in records
    }
    properties["inapplicable_concerns"] = {
        "type": "array",
        "maxItems": max(1, len(records)),
        "items": {
            "type": "object",
            "properties": {
                "concern": {"type": "string", "enum": list(records)},
                "reason": {"type": "string", "minLength": 1, "maxLength": 512},
            },
            "required": ["concern", "reason"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = [
    "StateSymbolTable",
    "compile_state_expression",
    "compile_state_mutations",
    "expression_schema",
    "mutation_schema",
    "render_state_model_concern",
    "state_concern_schema",
    "state_section_schema",
    "transition_schema",
    "validate_mutation_ir",
    "validate_state_concern",
    "validate_state_expr_ir",
    "validate_state_section",
    "variable_schema",
]
