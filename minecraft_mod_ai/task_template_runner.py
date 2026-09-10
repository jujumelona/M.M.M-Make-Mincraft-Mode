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


def execute_artifact_template(
    job: Any,
    context: dict[str, Any] | None = None,
    *,
    router: Any = None,
    port_registry: Any = None,
) -> dict[str, Any]:
    """Execute one leaf implementation template deterministically from exact mold and ports."""
    from .task_template_catalog import load_template
    from .implementation_template_renderer import render_template
    from .artifact_ports import PortKind, TypedPort
    from .artifact_validators import (
        validate_java_fragment,
        validate_json_resource,
        validate_registry_identifier,
    )
    from .atomic_slot_executor import fill_one_slot

    template_id = job.template_id if hasattr(job, "template_id") else job["template_id"]
    template = load_template(template_id)
    context_map = dict(context or {})
    det_inputs = dict(getattr(job, "deterministic_inputs", None) or (job.get("deterministic_inputs") if isinstance(job, dict) else {}) or {})
    values: dict[str, Any] = {**context_map, **det_inputs}

    # Resolve required inputs and ports
    requires = template.get("requires", ())
    for req in requires:
        if req not in values:
            if port_registry is not None and port_registry.has(req):
                values[req] = port_registry.get(req).value
            else:
                raise ValueError(
                    f"TEMPLATE_MISSING_REQUIREMENT: Required input {req!r} not found for {template.get('id')}"
                )

    # Resolve AI slots if any
    for slot in template.get("ai_slots", ()):
        slot_id = slot.get("id") or slot.get("slot_id")
        if slot_id not in values:
            values[slot_id] = fill_one_slot(router, slot, values)

    # Render output
    rendered_output = render_template(template, values)
    if hasattr(job, "rendered_output"):
        job.rendered_output = rendered_output

    # Render target and anchor
    target_spec = template.get("target") or {}
    target_file = ""
    anchor = ""
    if isinstance(target_spec, dict):
        if "file" in target_spec:
            target_file = render_template({"render": target_spec["file"]}, values)
        if "anchor" in target_spec:
            anchor = render_template({"render": target_spec["anchor"]}, values)
    if not target_file and hasattr(job, "target_path"):
        target_file = job.target_path
    if not anchor and hasattr(job, "anchor"):
        anchor = job.anchor

    # Run declared validators
    validation_receipts = []
    for validator_name in template.get("validators", ()):
        if validator_name == "java_parse":
            receipt = validate_java_fragment(rendered_output, anchor=anchor)
            validation_receipts.append(receipt)
        elif validator_name == "registry_identifier_unique":
            reg_id = f"{values.get('mod_id', 'mod')}:{values.get('registry_path', '')}"
            receipt = validate_registry_identifier(reg_id)
            validation_receipts.append(receipt)
        elif validator_name == "json_parse":
            receipt = validate_json_resource(rendered_output)
            validation_receipts.append(receipt)

    if hasattr(job, "validation_receipts"):
        job.validation_receipts = validation_receipts

    # Publish produced ports
    ports_published: dict[str, Any] = {}
    for produced in template.get("produces", ()):
        target_type = "Item"
        port_kind = PortKind.REGISTRY_ID
        val = ""
        if "registry_id" in produced:
            port_kind = PortKind.REGISTRY_ID
            val = f"{values.get('mod_id')}:{values.get('registry_path')}"
        elif "symbol" in produced:
            port_kind = PortKind.JAVA_SYMBOL
            val = f"ModItems.{values.get('java_constant')}"
        elif "model_ref" in produced:
            port_kind = PortKind.MODEL_REF
            val = f"{values.get('mod_id')}:item/{values.get('registry_path')}"
        elif "translation_key" in produced:
            port_kind = PortKind.TRANSLATION_KEY
            val = f"item.{values.get('mod_id')}.{values.get('registry_path')}"
        else:
            val = str(values.get(produced, ""))

        port_obj = TypedPort(name=produced, port_kind=port_kind, target_type=target_type, value=val)
        ports_published[produced] = port_obj
        if port_registry is not None:
            port_registry.publish(port_obj)

    receipt = {
        "status": "PASS",
        "template_id": template.get("id"),
        "target_file": target_file,
        "anchor": anchor,
        "rendered_output": rendered_output,
        "validations": validation_receipts,
        "ports_published": {k: p.to_dict() for k, p in ports_published.items()},
    }
    if hasattr(job, "status"):
        job.status = "SUCCESS"
    return receipt
