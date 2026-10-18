"""Generate independently required content properties as schema-closed grouped pages.

The host, not the model, owns property identities, ordering and typed validation.
Each field retains its exact target property schema (enums/numeric patterns).
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from jsonschema import Draft202012Validator

from .design_generation_schema import context_bound_record_schema
from .fixed_template_generation import generate_fixed_template_value
from .model_output_atomicity_contract import structured_output_token_ceiling
from .task_template_input import task_binding


def _page_schema(template: Mapping, normalized: Mapping, names: Sequence[str]) -> dict:
    properties = {}
    for name in names:
        bound = context_bound_record_schema(
            "design/content_property", template["record_schema"],
            {**normalized, "requested_property": name, "allowed_properties": [name]},
        )
        properties[name] = bound["properties"]["value"]
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(names)}


def _pages(template: Mapping, context: Mapping, required: Sequence[str]):
    from .execution_contract_policy import PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING

    offset = 0
    while offset < len(required):
        page = [required[offset]]
        schema = _page_schema(template, context, page)
        # Do not drop even a single field if its verified enum/pattern is large.
        structured_output_token_ceiling(schema)
        while offset + len(page) < len(required):
            candidate = [*page, required[offset + len(page)]]
            trial = _page_schema(template, context, candidate)
            if (structured_output_token_ceiling(trial, absolute_ceiling=None)
                    > PLANNER_RECORD_PAGE_OUTPUT_TOKEN_CEILING):
                break
            page, schema = candidate, trial
        yield page, schema
        offset += len(page)


def batched_property_records(router, template: Mapping, context: Mapping,
                             required: Sequence[str], progress, checkpoint):
    records = []
    for page, schema in _pages(template, context, required):
        binding_context = {**context, "required_properties": list(required),
                           "allowed_properties": list(page)}
        binding = "property-values-page-v1:" + task_binding(template, binding_context, ())
        value = (progress or {}).get(binding)
        if value is None:
            value = generate_fixed_template_value(
                router, "planner",
                (
                    {"role": "system", "content": (
                        template["task"] + "\n" + "\n".join(template["rules"]) + "\n"
                        "Return exactly one value for every schema-named property. "
                        "Property keys are immutable, host-selected identities. "
                        "Do not invent property names, resource IDs, or unrelated "
                        "Minecraft mechanics. Respect exact allowed values and "
                        "numeric formats of each property."
                    )},
                    {"role": "user", "content": json.dumps(
                        {"context": context, "requested_properties": page},
                        ensure_ascii=False, default=str)},
                ),
                response_schema=schema, enable_tools=False,
                description="Author independently required content properties.",
                output_token_ceiling=structured_output_token_ceiling(schema),
            )
            Draft202012Validator(schema).validate(value)
            if checkpoint is not None:
                checkpoint(binding, dict(value))
        Draft202012Validator(schema).validate(value)
        records.extend({"property": name, "value": value[name]} for name in page)
    return records
