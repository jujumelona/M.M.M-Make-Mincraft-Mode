"""Host-owned execution for one logical record without arbitrary field paging."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .design_generation_schema import context_bound_record_schema
from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import (
    assert_atomic_model_schema,
    effective_model_transport_schema,
    structured_output_token_ceiling,
)
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
            "different ordinal and do not make any continuation or completion decision. "
            "accepted_records contains the host-validated earlier records in this set. "
            "Choose the next distinct, still-uncovered authored entity or obligation. "
            "Do not repeat an accepted record, paraphrase the same entity into a new "
            "record, or add an ordinal/suffix to disguise a duplicate."
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
    """Batch up to three small related fields into one bounded model request.

    The 9B model authors a small complete record page rather than answering
    once per field. The host still proves each closed schema's token ceiling
    and validates the entire resulting record after every page.
    """

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
    undeclared_required = set(required).difference(properties)
    if undeclared_required:
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_SCHEMA: {identifier} has undeclared required fields "
            f"{sorted(undeclared_required)!r}"
        )
    shared = {key: deepcopy(schema[key]) for key in ("$schema", "$defs", "definitions")
              if key in schema}

    # A fixed small limit makes each page tractable even for an inexpensive
    # local model; no inference is spent asking whether to continue.
    limit = 3
    pages: list[dict[str, Any]] = []

    def make_page(fields: list[str]) -> dict[str, Any]:
        part_schema: dict[str, Any] = {
            **deepcopy(shared),
            "type": "object",
            "properties": {},
            "required": list(fields),
            "additionalProperties": False,
        }
        for name in fields:
            child = properties.get(name)
            if not isinstance(child, Mapping):
                raise SingleRecordTemplateError(
                    f"SINGLE_TEMPLATE_SCHEMA: {identifier}.{name} must declare a field schema"
                )
            part_schema["properties"][name] = deepcopy(dict(child))
        assert_atomic_model_schema(
            part_schema, surface=f"record page {identifier}: {','.join(fields)}"
        )
        structured_output_token_ceiling(effective_model_transport_schema(part_schema))
        return part_schema

    pending: list[str] = []
    for raw_name in required:
        name = str(raw_name)
        try:
            candidate = make_page([*pending, name])
        except Exception as exc:
            if not pending:
                raise SingleRecordTemplateError(
                    f"SINGLE_TEMPLATE_SCHEMA: unbounded field for {identifier}.{name}: {exc}"
                ) from exc
            pages.append(make_page(pending))
            pending = [name]
            try:
                make_page(pending)
            except Exception as inner:
                raise SingleRecordTemplateError(
                    f"SINGLE_TEMPLATE_SCHEMA: unbounded field for {identifier}.{name}: {inner}"
                ) from inner
        else:
            pending.append(name)
            if len(pending) >= limit:
                pages.append(candidate)
                pending = []
    if pending:
        pages.append(make_page(pending))
    if not pages:
        raise SingleRecordTemplateError(
            f"SINGLE_TEMPLATE_SCHEMA: {identifier} has no required model fields"
        )
    return tuple(pages)


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


def _deterministic_record_slice(
    schema: dict[str, Any], native_schema: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve a page without the model only when *all* fields are forced."""

    properties = native_schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    part: dict[str, Any] = {}
    for field in schema["required"]:
        native_field = properties.get(field)
        if not isinstance(native_field, Mapping):
            return None
        if "const" in native_field:
            candidate = native_field["const"]
        else:
            choices = native_field.get("enum")
            if not isinstance(choices, list) or len(choices) != 1:
                return None
            candidate = choices[0]
        part[field] = deepcopy(candidate)
    Draft202012Validator(schema).validate(part)
    return part

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
    schema = context_bound_record_schema(
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
                "content": json.dumps(normalized_context, ensure_ascii=False),
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
                            + ". Fill only the fields declared by the supplied response "
                            "schema and keep them semantically consistent with the accepted fields."
                        ),
                    }
                )
            tool_name = "submit_one_" + identifier.replace("/", "_")
            if len(slices) > 1:
                tool_name += f"_part_{part_index}_of_{len(slices)}"
            native_schema = effective_model_transport_schema(part_schema)
            part = _deterministic_record_slice(part_schema, native_schema)
            print(
                "PLANNER_RECORD_FIELD: " + json.dumps({
                    "template": identifier,
                    "fields": list(part_schema["required"]),
                    "field": ",".join(part_schema["required"]),
                    "generation": "deterministic" if part is not None else "model",
                }, ensure_ascii=False),
                flush=True,
            )
            if part is None:
                output_ceiling = structured_output_token_ceiling(native_schema)
                part = generate(
                    router,
                    "planner",
                    messages,
                    response_schema=part_schema,
                    enable_tools=False,
                    tool_name=tool_name,
                    output_token_ceiling=output_ceiling,
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
