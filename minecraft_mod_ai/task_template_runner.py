"One declared concern per model call, plus deterministic executable leaf templates."

from collections.abc import Mapping
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

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
        return any(
            _contains_blank_string(item, schema.get("properties", {}).get(key))
            for key, item in value.items()
        )
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
            "evidence_refs": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
        "required": ["status", "record", "reason", "evidence_refs"],
        "additionalProperties": False,
    }


def run_record_template(
    router, identifier, *, context, allowed_refs, progress=None, checkpoint=None
):
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
        value = (
            deepcopy(saved[len(accepted)])
            if replaying
            else generate_fixed_template_value(
                router,
                "planner",
                [
                    {
                        "role": "system",
                        "content": template["task"] + "\n" + "\n".join(template["rules"]),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                **context,
                                "accepted_records": records,
                                "allowed_evidence_refs": sorted(allowed_refs),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                response_schema=schema,
                enable_tools=False,
                tool_name="submit_" + identifier.replace("/", "_"),
            )
        )
        validator.validate(value)
        if any(ref not in allowed_refs for ref in value["evidence_refs"]):
            raise ValueError(f"TEMPLATE_EVIDENCE: unknown evidence in {identifier}")
        status = value["status"]
        record = value["record"]
        reason = value["reason"].strip()
        if status == "record":
            if record is None or _contains_blank_string(record, template["record_schema"]):
                raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
            key = json.dumps(record, sort_keys=True, ensure_ascii=False)
            if key in seen:
                raise TemplateBlocked(
                    f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}"
                )
            seen.add(key)
            records.append(record)
            refs.extend(ref for ref in value["evidence_refs"] if ref not in refs)
        elif record is not None:
            raise ValueError(f"TEMPLATE_STATUS: {status} cannot carry a record")
        elif status == "blocked":
            raise TemplateBlocked(
                f"TEMPLATE_BLOCKED: {identifier}: {reason or 'missing blocking reason'}"
            )
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
            return {
                "records": records,
                "reason": reason if status == "not_applicable" else "",
                "evidence_refs": refs,
            }


def _job_value(job: Any, name: str, default: Any):
    if hasattr(job, name):
        return getattr(job, name)
    if isinstance(job, dict):
        return job.get(name, default)
    return default


def _bind_job_dependencies(job: Any, values: dict[str, Any], port_registry: Any) -> None:
    dependencies = tuple(_job_value(job, "requires", ()) or ())
    if not dependencies:
        return
    if port_registry is None:
        raise ValueError(
            "TEMPLATE_PORT_REGISTRY_REQUIRED: job declares dependencies but no port registry was supplied"
        )
    for dependency in dependencies:
        port = port_registry.get(dependency)
        if port is None:
            raise ValueError(
                f"TEMPLATE_JOB_DEPENDENCY_MISSING: required scoped port {dependency!r} is unavailable"
            )
        alias = dependency.rsplit(".", 1)[-1]
        existing = values.get(alias)
        if existing is not None and str(existing) != port.value:
            raise ValueError(
                f"TEMPLATE_JOB_DEPENDENCY_CONFLICT: {dependency!r} resolves to {port.value!r} "
                f"but input {alias!r} already has {existing!r}"
            )
        values[alias] = port.value


def _logical_port(
    logical_name: str | Mapping[str, Any],
    published_name: str,
    values: dict[str, Any],
    template_id: str,
):
    from .artifact_ports import PortKind, TypedPort
    from .implementation_template_renderer import render_template

    if isinstance(logical_name, Mapping):
        kind_str = str(logical_name.get("kind") or "GENERIC")
        kind = PortKind(kind_str) if kind_str in PortKind.__members__.values() else PortKind.GENERIC
        target_type = str(logical_name.get("target_type") or "Object")
        raw_val = logical_name.get("value", "")
        if isinstance(raw_val, str) and "{{" in raw_val:
            rendered_val = render_template({"render": raw_val}, values)
        else:
            rendered_val = str(raw_val)
        return TypedPort(published_name, kind, target_type, rendered_val)

    mod_id = str(values.get("mod_id") or "")
    registry_path = str(values.get("registry_path") or "")
    constant = str(values.get("java_constant") or "")
    if logical_name == "item_key_symbol":
        return TypedPort(
            published_name, PortKind.JAVA_SYMBOL, "ResourceKey<Item>", f"ModItemIds.{constant}_KEY"
        )
    if logical_name == "item_registry_id":
        return TypedPort(
            published_name, PortKind.REGISTRY_ID, "Item", f"{mod_id}:{registry_path}"
        )
    if logical_name == "item_symbol":
        return TypedPort(
            published_name, PortKind.JAVA_SYMBOL, "Item", f"ModItems.{constant}"
        )
    if logical_name == "model_ref":
        return TypedPort(
            published_name,
            PortKind.MODEL_REF,
            "Item",
            f"{mod_id}:item/{registry_path}",
        )
    if logical_name == "translation_key":
        return TypedPort(
            published_name,
            PortKind.TRANSLATION_KEY,
            "Item",
            f"item.{mod_id}.{registry_path}",
        )
    if logical_name == "texture_ref":
        return TypedPort(
            published_name,
            PortKind.TEXTURE_REF,
            "Item",
            f"{mod_id}:item/{registry_path}",
        )
    if logical_name == "client_item_ref":
        return TypedPort(
            published_name,
            PortKind.CLIENT_ITEM_REF,
            "Item",
            f"{mod_id}:items/{registry_path}",
        )
    if logical_name == "block_key_symbol":
        return TypedPort(
            published_name, PortKind.JAVA_SYMBOL, "ResourceKey<Block>", f"ModBlockIds.{constant}_KEY"
        )
    if logical_name == "block_registry_id":
        return TypedPort(
            published_name, PortKind.REGISTRY_ID, "Block", f"{mod_id}:{registry_path}"
        )
    if logical_name == "block_symbol":
        return TypedPort(
            published_name, PortKind.JAVA_SYMBOL, "Block", f"ModBlocks.{constant}"
        )
    if logical_name == "blockstate_ref":
        return TypedPort(
            published_name, PortKind.MODEL_REF, "Block", f"{mod_id}:block/{registry_path}"
        )
    if logical_name == "block_model_ref":
        return TypedPort(
            published_name, PortKind.MODEL_REF, "Block", f"{mod_id}:block/{registry_path}"
        )
    if logical_name == "block_translation_key":
        return TypedPort(
            published_name, PortKind.TRANSLATION_KEY, "Block", f"block.{mod_id}.{registry_path}"
        )
    if logical_name == "block_loot_table_ref":
        return TypedPort(
            published_name, PortKind.GENERIC, "LootTable", f"{mod_id}:blocks/{registry_path}"
        )
    raise ValueError(
        f"TEMPLATE_PORT_UNDECLARED_SEMANTICS: template {template_id!r} publishes "
        f"unknown logical port {logical_name!r}"
    )


def execute_artifact_template(
    job: Any,
    context: dict[str, Any] | None = None,
    *,
    router: Any = None,
    port_registry: Any = None,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    from .artifact_validators import (
        validate_java_fragment,
        validate_json_resource,
        validate_registry_identifier,
    )
    from .atomic_slot_executor import fill_one_slot
    from .implementation_template_renderer import render_template
    from .task_template_catalog import load_template

    template_id = str(_job_value(job, "template_id", ""))
    if not template_id:
        raise ValueError("TEMPLATE_JOB_ID: artifact job has no template_id")
    template = load_template(template_id)
    if "render" not in template:
        raise ValueError(
            f"TEMPLATE_NOT_EXECUTABLE: {template_id!r} has no deterministic render mold"
        )

    context_map = dict(context or {})
    det_inputs = dict(_job_value(job, "deterministic_inputs", {}) or {})
    values: dict[str, Any] = {**context_map, **det_inputs}

    _bind_job_dependencies(job, values, port_registry)
    for required in tuple(template.get("requires", ()) or ()):
        if required not in values:
            raise ValueError(
                f"TEMPLATE_MISSING_REQUIREMENT: required input {required!r} "
                f"is missing for {template_id!r}"
            )

    for slot in tuple(template.get("ai_slots", ()) or ()):
        if not isinstance(slot, dict):
            raise ValueError(f"TEMPLATE_AI_SLOT_INVALID: {template_id!r} has a non-object slot")
        slot_id = slot.get("id") or slot.get("slot_id")
        if not slot_id:
            raise ValueError(f"TEMPLATE_AI_SLOT_ID: {template_id!r} has an unnamed slot")
        if slot_id not in values:
            values[slot_id] = fill_one_slot(router, slot, values)

    rendered_output = render_template(template, values)
    if hasattr(job, "rendered_output"):
        job.rendered_output = rendered_output

    target_spec = template.get("target") or {}
    target_file = str(_job_value(job, "target_path", "") or "")
    anchor = str(_job_value(job, "anchor", "") or "")
    if isinstance(target_spec, dict):
        if target_spec.get("file"):
            target_file = render_template({"render": target_spec["file"]}, values)
        if target_spec.get("anchor"):
            anchor = render_template({"render": target_spec["anchor"]}, values)

    validation_receipts = []
    supported_validators = {
        "java_parse",
        "registry_identifier_unique",
        "registry_identifier",
        "json_parse",
    }
    declared_validators = tuple(template.get("validators", ()) or ())
    unknown = [name for name in declared_validators if name not in supported_validators]
    if unknown:
        raise ValueError(
            f"TEMPLATE_VALIDATOR_UNKNOWN: {template_id!r} declares unsupported validators {unknown}"
        )
    for validator_name in declared_validators:
        if validator_name == "java_parse":
            validation_receipts.append(
                validate_java_fragment(rendered_output, anchor=anchor)
            )
        elif validator_name in {"registry_identifier_unique", "registry_identifier"}:
            validation_receipts.append(
                validate_registry_identifier(
                    f"{values.get('mod_id', '')}:{values.get('registry_path', '')}"
                )
            )
        elif validator_name == "json_parse":
            validation_receipts.append(validate_json_resource(rendered_output))

    if hasattr(job, "validation_receipts"):
        job.validation_receipts = validation_receipts

    logical_outputs = tuple(template.get("produces", ()) or ())
    scoped_outputs = tuple(_job_value(job, "produces", ()) or ())
    if scoped_outputs and len(scoped_outputs) != len(logical_outputs):
        raise ValueError(
            f"TEMPLATE_PORT_ARITY: job {str(_job_value(job, 'job_id', ''))!r} declares "
            f"{len(scoped_outputs)} scoped outputs for {len(logical_outputs)} template outputs"
        )
    published_names = scoped_outputs or logical_outputs

    ports_published: dict[str, Any] = {}
    for logical_name, published_name in zip(logical_outputs, published_names, strict=True):
        port_obj = _logical_port(logical_name, published_name, values, template_id)
        ports_published[published_name] = port_obj
        if port_registry is not None:
            port_registry.publish(port_obj)

    effective_base_dir = (
        base_dir or context_map.get("project_root") or context_map.get("base_dir")
    )
    materialization_data = None
    if effective_base_dir:
        from .artifact_materializer import materialize_job_output

        mat_receipt = materialize_job_output(
            job, rendered_output, base_dir=effective_base_dir
        )
        materialization_data = {
            "status": mat_receipt.status,
            "path": mat_receipt.target_path,
            "target_path": mat_receipt.target_path,
            "operation": mat_receipt.operation,
            "before_sha256": mat_receipt.before_sha256,
            "after_sha256": mat_receipt.after_sha256,
            "details": mat_receipt.details,
        }

    receipt = {
        "status": "PASS",
        "job_id": str(_job_value(job, "job_id", "")),
        "template_id": template_id,
        "target_file": target_file,
        "anchor": anchor,
        "rendered_output": rendered_output,
        "validations": validation_receipts,
        "materialization": materialization_data,
        "ports_published": {name: port.to_dict() for name, port in ports_published.items()},
    }
    if hasattr(job, "status"):
        job.status = "SUCCESS"
    return receipt
