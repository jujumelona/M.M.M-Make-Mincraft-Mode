"""One declared concern, one record per model call, deterministic completion."""
import json
from copy import deepcopy

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context


class TemplateBlocked(ValueError):
    pass


def _contains_blank_string(value, schema=None):
    schema = schema or {}
    if isinstance(value, str):
        return not value.strip() and schema.get("minLength", 1) > 0
    if isinstance(value, dict):
        return any(_contains_blank_string(item, schema.get("properties", {}).get(key))
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_blank_string(item, schema.get("items")) for item in value)
    return False


def record_response_schema(template):
    return {
        "type": "object",
        "properties": {
            "status": {"enum": ["record", "done", "not_applicable", "blocked"]},
            "record": {"anyOf": [template["record_schema"], {"type": "null"}]},
            "reason": {"type": "string"},
            "evidence_refs": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
        },
        "required": ["status", "record", "reason", "evidence_refs"],
        "additionalProperties": False,
    }


def run_record_template(router, identifier, *, context, allowed_refs, progress=None, checkpoint=None):
    template = load_record_template(identifier)
    context = task_context(template, context)
    schema = record_response_schema(template)
    validator = Draft202012Validator(schema)
    records, refs, seen = [], [], set()
    binding = task_binding(template, context, allowed_refs)
    saved = (progress or {}).get(binding, [])
    if not isinstance(saved, list):
        raise ValueError("TEMPLATE_PROGRESS: expected response array")
    accepted = []
    while True:
        replaying = len(accepted) < len(saved)
        value = deepcopy(saved[len(accepted)]) if replaying else generate_fixed_template_value(
            router, "planner",
            [{"role": "system", "content": template["task"] + "\n" + "\n".join(template["rules"])},
             {"role": "user", "content": json.dumps({
                 **context, "accepted_records": records,
                 "allowed_evidence_refs": sorted(allowed_refs),
             }, ensure_ascii=False)}],
            response_schema=schema, enable_tools=False,
            tool_name="submit_" + identifier.replace("/", "_"),
        )
        validator.validate(value)
        if any(ref not in allowed_refs for ref in value["evidence_refs"]):
            raise ValueError(f"TEMPLATE_EVIDENCE: unknown evidence in {identifier}")
        status, record, reason = value["status"], value["record"], value["reason"].strip()
        if status == "record":
            if record is None or _contains_blank_string(record, template["record_schema"]):
                raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
            key = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if key in seen:
                raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}")
            seen.add(key)
            records.append(record)
            refs.extend(ref for ref in value["evidence_refs"] if ref not in refs)
        elif record is not None:
            raise ValueError(f"TEMPLATE_STATUS: {status} cannot carry a record")
        elif status == "blocked":
            raise TemplateBlocked(f"TEMPLATE_BLOCKED: {identifier}: {reason or 'missing blocking reason'}")
        elif status == "done" and records:
            refs.extend(ref for ref in value["evidence_refs"] if ref not in refs)
        elif status == "not_applicable" and not records and reason:
            refs = value["evidence_refs"]
        else:
            raise ValueError(f"TEMPLATE_STATUS: invalid {status} transition in {identifier}")
        accepted.append(deepcopy(value))
        if status != "record" and len(accepted) < len(saved):
            raise ValueError("TEMPLATE_PROGRESS: responses after completion")
        if not replaying and checkpoint is not None:
            checkpoint(binding, deepcopy(accepted))
        if status != "record":
            return {"records": records, "reason": reason if status == "not_applicable" else "", "evidence_refs": refs}
