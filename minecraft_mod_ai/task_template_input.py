"""Declared task inputs and content-bound checkpoint identity."""
import json
from copy import deepcopy
from hashlib import sha256

from jsonschema import Draft202012Validator

_HOST_CONTROL_KEYS = frozenset({
    "accepted_records",
    "record_index",
    "record_count",
    "record_ordinal",
    "requested_property",
    "entity_ordinal",
    "entity_count",
    "source_id",
    "target_id",
    "allowed_relation_types",
})


def task_context(template, context):
    schema = template.get("input_schema")
    if schema is None:
        return deepcopy(context)
    semantic = {
        key: deepcopy(context[key])
        for key in schema["properties"]
        if key in context
    }
    Draft202012Validator(schema).validate(semantic)
    controls = {
        key: deepcopy(context[key])
        for key in _HOST_CONTROL_KEYS
        if key in context and key not in semantic
    }
    return {**semantic, **controls}


def task_binding(template, context, allowed_refs=()):
    return sha256(json.dumps(
        [template, context, sorted(allowed_refs)], sort_keys=True,
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")).hexdigest()
