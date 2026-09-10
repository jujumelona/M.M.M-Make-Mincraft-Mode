"""Host-owned execution for templates whose cardinality is already known."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context


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


def run_single_record_template(
    router: Any,
    identifier: str,
    *,
    context: dict[str, Any],
    progress: dict[str, Any] | None = None,
    checkpoint: Any = None,
) -> dict[str, Any]:
    """Generate exactly one record; the model never owns iteration or completion."""

    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    schema = template["record_schema"]
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
        value = generate_fixed_template_value(
            router,
            "planner",
            [
                {
                    "role": "system",
                    "content": template["task"] + "\n" + "\n".join(template["rules"]),
                },
                {
                    "role": "user",
                    "content": json.dumps(normalized_context, ensure_ascii=False),
                },
            ],
            response_schema=schema,
            enable_tools=False,
            tool_name="submit_one_" + identifier.replace("/", "_"),
        )
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
