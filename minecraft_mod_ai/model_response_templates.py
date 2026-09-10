"""Fixed response contracts for model calls outside the engineering worksheet."""
from copy import deepcopy
import json


def _object(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items, "maxItems": 4}


_TEXT = {"type": "string", "maxLength": 256}
_NONEMPTY = {"type": "string", "minLength": 1, "maxLength": 256}
_STATUS = {"type": "string", "enum": ["PASS", "FAIL"]}
_REPLACEMENT = _object({"old": _NONEMPTY, "new": _TEXT,
                        "count": {"type": "integer", "minimum": 1}})
_PATCH_OPERATIONS = []
for _kind in ("create", "replace", "edit"):
    _fields = {"operation": {"type": "string", "enum": [_kind]}, "path": _NONEMPTY}
    if _kind != "create":
        _fields["expected_sha256"] = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$", "maxLength": 71}
    if _kind == "edit":
        _fields["replacements"] = {**_array(_REPLACEMENT), "minItems": 1}
    else:
        _fields["content"] = _TEXT
    _PATCH_OPERATIONS.append(_object(_fields))

_TEMPLATES = {
    "repair": _object({"operations": {**_array({"anyOf": _PATCH_OPERATIONS}), "minItems": 1}}),
    "visual_review": _object({
        "status": _STATUS,
        "findings": _array(_TEXT),
        "acceptance_test_results": _array(_object({
            "test": _NONEMPTY, "status": _STATUS, "evidence": _NONEMPTY,
        })),
    }),
    "capabilities": _object({"capabilities": _array(_object({
        "capability_id": _NONEMPTY, "source_span": _NONEMPTY,
        "description": _NONEMPTY, "category": _NONEMPTY,
        "dependencies": _array(_NONEMPTY),
    }))}),
    "coder_summary": _object({"summary": _TEXT}),
}


def response_schema(name):
    return deepcopy(_TEMPLATES[name])


def response_template_prompt(name):
    return (
        "Return exactly one JSON value that conforms to this fixed response schema. "
        "Do not return the schema itself. Do not invent, rename, or omit fields. "
        "Populate only the allowed value slots: "
        + json.dumps(response_schema(name), ensure_ascii=False, separators=(",", ":"))
    )
