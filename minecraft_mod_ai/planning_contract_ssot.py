from __future__ import annotations

"""Single Source of Truth (SSOT) for all planning model contracts and templates.

Every model-authored schema in the planning pipeline is defined or validated here.
All schemas are strictly closed (additionalProperties=False) and satisfy
the global model output atomicity bounds.

At module import time, `assert_all_planning_contracts_valid()` verifies every schema
in the system so contradictory or open schemas can never be introduced.
"""

from collections.abc import Mapping
from typing import Any

from .model_output_atomicity_contract import (
    _assert_closed_object_schemas,
    assert_atomic_model_schema,
)
from .planning_detail_slots import DETAIL_RECORDS

# The host assembly schema is derived from task files, never supplied as one model call.
from .task_template_catalog import load_template

MODEL_UNRESOLVED_REASONS = tuple(load_template("prompt/ambiguities")["record_schema"]["properties"]["reason"]["enum"])
PROMPT_STATE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": load_template("prompt/intent")["output_schema"],
        "known": {"type": "array", "items": load_template("prompt/parse")["record_schema"]},
        "references": {"type": "array", "items": load_template("prompt/entity_resolution")["record_schema"]},
        "scope_status": load_template("prompt/scope")["output_schema"]["properties"]["scope_status"],
        "unresolved": {"type": "array", "items": load_template("prompt/ambiguities")["record_schema"]},
    },
    "required": ["goal", "known", "references", "scope_status", "unresolved"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# 2. submit_researched_requirements contract
# ---------------------------------------------------------------------------
SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "maxItems": 4,
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "maxLength": 256},
                    "semantic_capability": {"type": "string", "maxLength": 256},
                    "acceptance": {"type": "string", "maxLength": 256},
                },
                "required": ["statement", "semantic_capability", "acceptance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["requirements"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# 3. research note contract
# ---------------------------------------------------------------------------
RESEARCH_NOTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "research_note": {
            "type": "object",
            "properties": {
                "domain_id": {"type": "string"},
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "evidence_ref": {"type": "string"},
                        },
                        "required": ["claim"],
                        "additionalProperties": False,
                    },
                },
                "gaps": {"type": "array", "items": {"type": "string"}},
                "next_queries": {"type": "array", "items": {"type": "string"}},
                "sufficient": {"type": "boolean"},
                "procedures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["domain_id", "claims", "gaps", "sufficient"],
            "additionalProperties": False,
        }
    },
    "required": ["research_note"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# 4. pre-design support contract
# ---------------------------------------------------------------------------
def pre_design_support_schema(count: int) -> dict[str, Any]:
    """Strictly closed schema for claim-support verification."""
    verdict_item = {
        "type": "object",
        "properties": {
            "claim_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": max(0, count - 1),
            },
            "supported": {"type": "boolean"},
            "support_quote": {"type": "string"},
            "quote": {"type": "string"},
            "support": {"type": "string"},
        },
        "required": ["claim_index"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": verdict_item,
            },
            "claims": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": verdict_item,
            },
            "sufficient": {"type": "boolean"},
            "gaps": {"type": "array", "items": {"type": "string"}},
            "research_note": {"type": "string"},
        },
        "required": ["claims"],
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# 5. Skeleton template renderer & anti-echo detection
# ---------------------------------------------------------------------------
def schema_skeleton_template(schema: Mapping[str, Any], *, max_depth: int = 10) -> Any:
    """Render a clean data skeleton template with field placeholders.

    This explicitly contains ZERO schema meta-keys (no 'type', 'properties',
    'required', or 'additionalProperties') so small models cannot be tricked
    into echoing the schema definition itself.
    """
    if not isinstance(schema, Mapping) or max_depth <= 0:
        return "<value>"

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            return {
                str(key): schema_skeleton_template(prop_schema, max_depth=max_depth - 1)
                for key, prop_schema in properties.items()
            }
        return {}

    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, Mapping):
            # Show exactly 1 sample item in the array skeleton
            return [schema_skeleton_template(items, max_depth=max_depth - 1)]
        return []

    if schema_type == "string":
        enum_values = schema.get("enum")
        if isinstance(enum_values, (list, tuple)) and enum_values:
            return " | ".join(str(v) for v in enum_values)
        desc = schema.get("description")
        if desc:
            return f"<{desc}>"
        return "<string>"

    if schema_type == "integer":
        return 0

    if schema_type == "number":
        return 0.0

    if schema_type == "boolean":
        return False

    return "<value>"


def is_schema_definition_echo(value: Any) -> bool:
    """Return True if the decoded JSON looks like a schema definition rather than data."""
    if not isinstance(value, Mapping):
        return False
    # If top-level contains 'type': 'object' and 'properties' or 'additionalProperties'
    if value.get("type") == "object" and ("properties" in value or "additionalProperties" in value):
        return True
    if "properties" in value and "$schema" in value:
        return True
    return False


# ---------------------------------------------------------------------------
# 6. Cross-contract self-verification
# ---------------------------------------------------------------------------
def assert_all_planning_contracts_valid() -> None:
    """Enforce that every planning schema passes its applicable model-contract checks."""
    from .worksheet_atomic_chunker import pack_section_concerns, worksheet_chunk_schema

    fixed_schemas: list[tuple[str, Mapping[str, Any]]] = [
        ("PROMPT_STATE_INPUT_SCHEMA", PROMPT_STATE_INPUT_SCHEMA),
        (
            "SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA",
            SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
        ),
        ("RESEARCH_NOTE_SCHEMA", RESEARCH_NOTE_SCHEMA),
        ("pre_design_support_schema(2)", pre_design_support_schema(2)),
    ]

    for name, schema in fixed_schemas:
        _assert_closed_object_schemas(schema, path=f"planning_contract:{name}")

    # This schema is passed directly to generate_tool_decision, so validate the
    # complete native-tool atomicity contract here rather than waiting for runtime.
    assert_atomic_model_schema(
        SUBMIT_RESEARCHED_REQUIREMENTS_SCHEMA,
        surface="planner native tool submit_researched_requirements",
    )

    # Validate all worksheet concern chunks across every section
    for section in DETAIL_RECORDS:
        chunks = pack_section_concerns(section)
        for index, concerns in enumerate(chunks, start=1):
            is_first = index == 1
            chunk_schema = worksheet_chunk_schema(
                section, concerns, include_evidence=is_first
            )
            surface = f"worksheet_chunk:{section}:{index}"
            _assert_closed_object_schemas(chunk_schema, path=surface)


# Run self-verification upon module load
assert_all_planning_contracts_valid()
