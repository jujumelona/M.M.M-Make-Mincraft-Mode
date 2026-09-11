"One declared concern per model call, plus deterministic executable leaf templates."

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .fixed_template_generation import generate_fixed_template_value
from .task_template_catalog import load_record_template
from .task_template_input import task_binding, task_context


class TemplateBlocked(ValueError):
    pass


_HOST_EXACT_SINGLE_TEMPLATES = frozenset({
    "design/content_capability",
})

_HOST_SEQUENCE_TEMPLATES = frozenset({
    "design/research_fact",
    "design/content_entity",
    "design/content_relation",
    "design/decision",
    "design/content_property",
})

_HOST_FIRST_RECORD_REQUIRED = frozenset({
    "design/content_entity",
})


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


def _grounded_evidence_refs(context, allowed_refs):
    grounded_refs: list[str] = []
    for key in ("source_evidence_id", "evidence_ref", "shard_id"):
        ref = context.get(key)
        if isinstance(ref, str) and ref in allowed_refs and ref not in grounded_refs:
            grounded_refs.append(ref)
    evidence = context.get("evidence")
    if isinstance(evidence, str) and evidence in allowed_refs and evidence not in grounded_refs:
        grounded_refs.append(evidence)
    elif isinstance(evidence, (list, tuple, set)):
        for ref in evidence:
            if isinstance(ref, str) and ref in allowed_refs and ref not in grounded_refs:
                grounded_refs.append(ref)
    return grounded_refs


def _accepted_record_index(identifier, records):
    """Return only the compact identity needed by the continuation decision."""
    if identifier == "design/content_entity":
        return [str(row.get("entity_id", "")) for row in records]
    if identifier == "design/content_relation":
        return [
            ":".join(
                str(row.get(key, ""))
                for key in ("relation_type", "source_id", "target_id")
            )
            for row in records
        ]
    if identifier == "design/decision":
        return [str(row.get("slot_id", "")) for row in records]
    if identifier == "design/content_property":
        return [str(row.get("property", "")) for row in records]
    return [
        json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for row in records
    ]


def _run_host_owned_records(
    router,
    identifier,
    *,
    context,
    allowed_refs,
    progress=None,
    checkpoint=None,
):
    """Run exact/sequence design records without giving the model a `done` control field."""
    from .single_record_template import run_single_record_template

    template = load_record_template(identifier)
    normalized_context = task_context(template, context)
    refs = _grounded_evidence_refs(normalized_context, allowed_refs)

    if identifier in _HOST_EXACT_SINGLE_TEMPLATES:
        record = run_single_record_template(
            router,
            identifier,
            context=normalized_context,
            progress=progress,
            checkpoint=checkpoint,
        )
        return {"records": [record], "reason": "", "evidence_refs": refs}

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    while True:
        if identifier in _HOST_FIRST_RECORD_REQUIRED and not records:
            required = True
        else:
            continuation = run_single_record_template(
                router,
                "design/continue_record",
                context={
                    "target_template": identifier,
                    "target_task": template["task"],
                    "target_rules": list(template.get("rules", ())),
                    "active_context": normalized_context,
                    "accepted_record_ids": _accepted_record_index(identifier, records),
                },
                progress=progress,
                checkpoint=checkpoint,
            )
            required = bool(continuation["required"])
        if not required:
            return {"records": records, "reason": "", "evidence_refs": refs}

        record_context = {
            **normalized_context,
            "accepted_records": deepcopy(records),
        }
        if identifier == "design/content_property":
            allowed_properties = normalized_context.get("allowed_properties")
            if isinstance(allowed_properties, list):
                used = {str(row.get("property", "")) for row in records}
                remaining = [name for name in allowed_properties if name not in used]
                if not remaining:
                    return {"records": records, "reason": "", "evidence_refs": refs}
                record_context["allowed_properties"] = remaining

        record = run_single_record_template(
            router,
            identifier,
            context=record_context,
            progress=progress,
            checkpoint=checkpoint,
        )
        if _contains_blank_string(record, template["record_schema"]):
            raise ValueError(f"TEMPLATE_RECORD: empty record in {identifier}")
        key = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if key in seen:
            raise TemplateBlocked(f"TEMPLATE_NO_PROGRESS: repeated record in {identifier}")
        seen.add(key)
        records.append(record)


def record_response_schema(template):
    return {
        "type": "object",
        "properties": {
            "status": {"enum": ["record", "done", "not_applicable", "blocked"]},
            "record": {"anyOf": [template["record_schema"], {"type": "null"}]},
            "reason": {"type": "string", "maxLength": 256},
        },
        "required": ["status", "record", "reason"],
        "additionalProperties": False,
    }


def run_record_template(
    router, identifier, *, context, allowed_refs, progress=None, checkpoint=None
):
    if identifier in _HOST_EXACT_SINGLE_TEMPLATES or identifier in _HOST_SEQUENCE_TEMPLATES:
        return _run_host_owned_records(
            router,
            identifier,
            context=context,
            allowed_refs=allowed_refs,
            progress=progress,
            checkpoint=checkpoint,
        )

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
        evidence_from_value = value.pop("evidence_refs", None)
        validator.validate(value)
        if evidence_from_value is not None:
            value["evidence_refs"] = list(evidence_from_value)
        else:
            value["evidence_refs"] = _grounded_evidence_refs(context, allowed_refs)
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
    required_types = {p["name"]: p for p in _job_value(job, "required_ports", ())}
    aliases = [str(d).rsplit(".", 1)[-1] for d in dependencies if isinstance(d, str)]
    for dependency in dependencies:
        if isinstance(dependency, Mapping) or hasattr(dependency, "port_kind"):
            name = getattr(dependency, "name", None) or dependency.get("name")
            kind = getattr(dependency, "port_kind", None) or dependency.get("kind")
            ttype = getattr(dependency, "target_type", None) or dependency.get("target_type")
            port = port_registry.resolve(name, kind, ttype)
            dep_name = name
        else:
            dep_name = str(dependency)
            port = port_registry.get(dep_name)
            if port is None:
                raise ValueError(
                    f"TEMPLATE_JOB_DEPENDENCY_MISSING: required scoped port {dep_name!r} is unavailable"
                )
        expected = required_types.get(dep_name)
        if expected is not None:
            port_registry.resolve(dep_name, expected["kind"], expected["target_type"])
        alias = dep_name.rsplit(".", 1)[-1]
        values.setdefault("dependency_ports", {})[dep_name] = port.value
        if aliases.count(alias) > 1:
            continue
        existing = values.get(alias)
        if existing is not None and str(existing) != port.value:
            raise ValueError(
                f"TEMPLATE_JOB_DEPENDENCY_CONFLICT: {dep_name!r} resolves to {port.value!r} "
                f"but input {alias!r} already has {existing!r}"
            )
        values[alias] = port.value


_STANDARD_PORT_DEFINITIONS: dict[str, tuple[str, str, str]] = {
    "item_key_symbol": ("JAVA_SYMBOL", "ResourceKey<Item>", "ModItemIds.{constant}_KEY"),
    "item_registry_id": ("REGISTRY_ID", "Item", "{mod_id}:{registry_path}"),
    "item_symbol": ("JAVA_SYMBOL", "Item", "ModItems.{constant}"),
    "model_ref": ("MODEL_REF", "Item", "{mod_id}:item/{registry_path}"),
    "translation_key": ("TRANSLATION_KEY", "Item", "item.{mod_id}.{registry_path}"),
    "texture_ref": ("TEXTURE_REF", "Item", "{mod_id}:item/{registry_path}"),
    "client_item_ref": ("CLIENT_ITEM_REF", "Item", "{mod_id}:items/{registry_path}"),
    "block_key_symbol": ("JAVA_SYMBOL", "ResourceKey<Block>", "ModBlockIds.{constant}_KEY"),
    "block_registry_id": ("REGISTRY_ID", "Block", "{mod_id}:{registry_path}"),
    "block_symbol": ("JAVA_SYMBOL", "Block", "ModBlocks.{constant}"),
    "blockstate_ref": ("MODEL_REF", "Block", "{mod_id}:block/{registry_path}"),
    "block_model_ref": ("MODEL_REF", "Block", "{mod_id}:block/{registry_path}"),
    "block_translation_key": ("TRANSLATION_KEY", "Block", "block.{mod_id}.{registry_path}"),
    "block_loot_table_ref": ("GENERIC", "LootTable", "{mod_id}:blocks/{registry_path}"),
    "entity_key_symbol": ("JAVA_SYMBOL", "ResourceKey<EntityType<?>>", "ModEntityIds.{constant}_KEY"),
    "entity_registry_id": ("REGISTRY_ID", "EntityType<?>", "{mod_id}:{registry_path}"),
    "entity_symbol": ("JAVA_SYMBOL", "EntityType<?>", "ModEntities.{constant}"),
    "entity_translation_key": ("TRANSLATION_KEY", "EntityType<?>", "entity.{mod_id}.{registry_path}"),
}


def _logical_port(
    logical_name: str | Mapping[str, Any],
    published_name: str,
    values: dict[str, Any],
    template_id: str,
):
    from .artifact_ports import PortKind, TypedPort
    from .implementation_template_renderer import render_template

    if isinstance(logical_name, Mapping):
        for key in ("name", "kind", "target_type", "value"):
            if not isinstance(logical_name.get(key), str) or not logical_name[key].strip():
                raise ValueError(f"TEMPLATE_PORT_DECLARATION: {template_id!r} missing {key}")
        kind = PortKind(logical_name["kind"])
        target_type = logical_name["target_type"]
        raw_val = logical_name["value"]
        if isinstance(raw_val, str) and "{{" in raw_val:
            rendered_val = render_template({"render": raw_val}, values)
        else:
            rendered_val = str(raw_val)
        return TypedPort(published_name, kind, target_type, rendered_val)

    if logical_name in _STANDARD_PORT_DEFINITIONS:
        kind_name, target_type, pattern = _STANDARD_PORT_DEFINITIONS[logical_name]
        kind = getattr(PortKind, kind_name)
        rendered_val = pattern.format(
            mod_id=str(values.get("mod_id") or ""),
            registry_path=str(values.get("registry_path") or ""),
            constant=str(values.get("java_constant") or ""),
        )
        return TypedPort(published_name, kind, target_type, rendered_val)
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

    from .artifact_target_contract import validate_artifact_target

    validate_artifact_target(template, values.get("minecraft_version", ""))
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
        "resource_references",
    }
    declared_validators = tuple(template.get("validators", ()) or ())
    unknown = [name for name in declared_validators if name not in supported_validators]
    if unknown:
        raise ValueError(
            f"TEMPLATE_VALIDATOR_UNKNOWN: {template_id!r} declares unsupported validators {unknown}"
        )
    for validator_name in declared_validators:
        if validator_name == "java_parse":
            validation_receipts.append(validate_java_fragment(rendered_output, anchor=anchor))
        elif validator_name in {"registry_identifier_unique", "registry_identifier"}:
            validation_receipts.append(
                validate_registry_identifier(
                    f"{values.get('mod_id', '')}:{values.get('registry_path', '')}"
                )
            )
        elif validator_name == "resource_references":
            from .artifact_validators.resource_links import validate_resource_links

            validation_receipts.append(
                validate_resource_links(rendered_output, values, port_registry)
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
    published_names = scoped_outputs or tuple(
        port["name"] if isinstance(port, Mapping) else port for port in logical_outputs
    )

    ports_published: dict[str, Any] = {}
    for logical_name, published_name in zip(logical_outputs, published_names, strict=True):
        port_obj = _logical_port(logical_name, published_name, values, template_id)
        if published_name in ports_published:
            raise ValueError(f"TEMPLATE_PORT_DUPLICATE: {published_name}")
        if port_registry is not None:
            existing_port = port_registry.get(published_name)
            if existing_port is not None and existing_port != port_obj:
                raise ValueError(f"TEMPLATE_PORT_CONFLICT: {published_name}")
        ports_published[published_name] = port_obj

    effective_base_dir = base_dir or context_map.get("project_root") or context_map.get("base_dir")
    materialization_data = None
    if effective_base_dir:
        from dataclasses import replace

        from .artifact_job import ArtifactJob
        from .artifact_materializer import materialize_job_output

        materialize_job = (
            replace(
                job,
                target_path=target_file,
                anchor=anchor,
                operation=target_spec.get("operation", _job_value(job, "operation", "")),
            )
            if isinstance(job, ArtifactJob)
            else job
        )
        if (
            isinstance(job, ArtifactJob)
            and job.operation
            and (job.target_path != target_file or job.anchor != anchor)
        ):
            raise ValueError(
                "TEMPLATE_JOB_TARGET_DRIFT: re-expand the job against the current template"
            )
        mat_receipt = materialize_job_output(
            materialize_job, rendered_output, base_dir=effective_base_dir
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

        target_p = Path(mat_receipt.target_path)
        if not target_p.exists():
            raise ValueError(
                f"TEMPLATE_POST_WRITE_FAILED: Target file {target_p} was not written"
            )

    if port_registry is not None:
        for port_obj in ports_published.values():
            port_registry.publish(port_obj)

    receipt = {
        "status": "PASS",
        "job_id": str(_job_value(job, "job_id", "")),
        "template_id": template_id,
        "target_file": target_file,
        "anchor": anchor,
        "rendered_output": rendered_output,
        "validations": validation_receipts,
        "materialization": materialization_data,
        "ports_published": {
            name: port.to_dict() for name, port in ports_published.items()
        },
    }
    if hasattr(job, "status"):
        job.status = "SUCCESS"
    return receipt
