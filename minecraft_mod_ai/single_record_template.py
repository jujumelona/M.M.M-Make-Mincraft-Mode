"""Host-owned execution for templates whose cardinality is already known."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable

from jsonschema import Draft202012Validator

from .design_generation_schema import context_bound_record_schema
from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import MAX_MODEL_FIELDS, assert_atomic_model_schema
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context


_READ_ONLY_CONTEXT_CONTRACT = (
    "The user message is READ_ONLY_INPUT_CONTEXT, not an output example. "
    "Use it only to derive the requested semantic values. Never echo, quote, serialize, "
    "or embed the context object, its JSON representation, field labels, schema text, "
    "prompt text, or tool protocol into an output field."
)


class SingleRecordTemplateError(ValueError):
    pass


def _context_bound_record_schema(
    identifier: str,
    schema: dict[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility seam for tests and callers using the former private helper."""
    return context_bound_record_schema(identifier, schema, context)


def _contains_blank_string(value: Any, schema: dict[str, Any] | None = None) -> bool:
    schema = schema or {}
    if isinstance(value, str):
        return not value.strip() and schema.get("minLength", 1) > 0
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        return any(
            _contains_blank_string(item, properties.get(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_blank_string(item, schema.get("items")) for item in value)
    return False


def _ordinal_instruction(context: dict[str, Any]) -> str:
    count = context.get("record_count")
    index = context.get("record_index")
    if type(count) is int and type(index) is int:
        return (
            f"\nThe host has established exactly {count} records. Return only record "
            f"{index + 1} of {count}, using stable source/authored order. Do not return a "
            "different ordinal and do not make any continuation or completion decision."
        )
    entity_count = context.get("entity_count")
    entity_ordinal = context.get("entity_ordinal")
    if type(entity_count) is int and type(entity_ordinal) is int:
        return (
            f"\nThe host has established exactly {entity_count} entities. Return only entity "
            f"{entity_ordinal} of {entity_count}, using stable source/authored order. Do not "
            "make any continuation or completion decision."
        )
    return ""


def _atomic_record_schema_slices(
    schema: dict[str, Any],
    *,
    identifier: str,
) -> tuple[dict[str, Any], ...]:
    """Project one logical record into deterministic small-model field slices."""
    if schema.get("type") != "object":
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_SCHEMA: {identifier} record_schema must be an object"
        )
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_SCHEMA: {identifier} must declare properties and required fields"
        )

    field_names = tuple(properties)
    if not field_names:
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_SCHEMA: {identifier} has no model-authored fields"
        )

    required_set = set(required)
    undeclared_required = required_set.difference(field_names)
    if undeclared_required:
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_SCHEMA: {identifier} has undeclared required fields "
            f"{sorted(undeclared_required)!r}"
        )

    slices: list[dict[str, Any]] = []
    for start in range(0, len(field_names), MAX_MODEL_FIELDS):
        names = field_names[start : start + MAX_MODEL_FIELDS]
        projected = {
            "type": "object",
            "properties": {name: deepcopy(properties[name]) for name in names},
            "required": [name for name in names if name in required_set],
            "additionalProperties": False,
        }
        for metadata_key in ("title", "description"):
            if metadata_key in schema:
                projected[metadata_key] = deepcopy(schema[metadata_key])
        try:
            assert_atomic_model_schema(
                projected,
                surface=f"record slice for {identifier!r}",
            )
        except Exception as exc:
            raise SingleRecordTemplateError(
                f"SINGLE_TEMPLATE_ATOMIC_PROJECTION: cannot project {identifier} fields "
                f"{list(names)!r} into one atomic model call: {exc}"
            ) from exc
        slices.append(projected)
    return tuple(slices)


def _merge_record_slice(
    merged: dict[str, Any],
    part: Any,
    *,
    identifier: str,
) -> None:
    if not isinstance(part, Mapping):
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_RECORD: expected object slice for {identifier}"
        )
    overlap = set(merged).intersection(part)
    if overlap:
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_RECORD: duplicate projected fields for {identifier}: "
            f"{sorted(overlap)!r}"
        )
    merged.update(deepcopy(dict(part)))


def run_single_record_template(
    router: Any,
    identifier: str,
    *,
    context: dict[str, Any],
    progress: dict[str, Any] | None = None,
    checkpoint: Any = None,
    generator: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Generate exactly one record; the model never owns iteration or completion."""
    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    schema = _context_bound_record_schema(
        identifier,
        template["record_schema"],
        normalized_context,
    )
    validator = Draft202012Validator(schema)
    binding = "single:" + task_binding(template, normalized_context, ())

    saved = (progress or {}).get(binding)
    if saved is not None:
        if not isinstance(saved, dict):
            raise SingleRecordTemplateError(
                f"SINGLE_TEMPLATE_PROGRESS: expected one object for {identifier}"
            )
        value = deepcopy(saved)
    else:
        generate = generator or generate_fixed_template_value
        base_messages = [
            {
                "role": "system",
                "content": (
                    template["task"]
                    + "\n"
                    + "\n".join(template["rules"])
                    + "\n"
                    + _READ_ONLY_CONTEXT_CONTRACT
                    + _ordinal_instruction(normalized_context)
                ),
            },
            {
                "role": "user",
                "content": (
                    "READ_ONLY_INPUT_CONTEXT:\n"
                    + json.dumps(normalized_context, ensure_ascii=False)
                ),
            },
        ]
        slices = _atomic_record_schema_slices(schema, identifier=identifier)
        value: dict[str, Any] = {}
        for part_index, part_schema in enumerate(slices, start=1):
            messages = list(base_messages)
            if value:
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "The host has already accepted these fields for the same logical "
                            "record: "
                            + json.dumps(value, ensure_ascii=False, sort_keys=True)
                            + ". Fill only the fields declared by the supplied native function "
                            "schema and keep them semantically consistent with the accepted fields."
                        ),
                    }
                )
            tool_name = "submit_one_" + identifier.replace("/", "_")
            if len(slices) > 1:
                tool_name += f"_part_{part_index}_of_{len(slices)}"
            part = generate(
                router,
                "planner",
                messages,
                response_schema=part_schema,
                enable_tools=False,
                tool_name=tool_name,
            )
            Draft202012Validator(part_schema).validate(part)
            _merge_record_slice(value, part, identifier=identifier)

        validator.validate(value)
        if _contains_blank_string(value, schema):
            raise SingleRecordTemplateError(
                f"SINGLE_TEMPLATE_RECORD: empty record in {identifier}"
            )
        if checkpoint is not None:
            checkpoint(binding, deepcopy(value))

    validator.validate(value)
    if _contains_blank_string(value, schema):
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_RECORD: empty record in {identifier}"
        )
    return value
