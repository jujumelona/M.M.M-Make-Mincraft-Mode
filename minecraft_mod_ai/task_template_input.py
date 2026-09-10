"""Declared task inputs and content-bound checkpoint identity."""
import json
from copy import deepcopy
from hashlib import sha256

from jsonschema import Draft202012Validator


def task_context(template, context):
    schema = template.get("input_schema")
    if schema is None:
        return deepcopy(context)
    selected = {key: deepcopy(context[key]) for key in schema["properties"] if key in context}
    Draft202012Validator(schema).validate(selected)
    return selected


def task_binding(template, context, allowed_refs=()):
    return sha256(json.dumps(
        [template, context, sorted(allowed_refs)], sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
