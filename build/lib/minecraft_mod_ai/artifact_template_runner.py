"""Deterministic executable artifact-template runtime."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


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
    "entity_key_symbol": (
        "JAVA_SYMBOL",
        "ResourceKey<EntityType<?>>",
        "ModEntityIds.{constant}_KEY",
    ),
    "entity_registry_id": ("REGISTRY_ID", "EntityType<?>", "{mod_id}:{registry_path}"),
    "entity_symbol": ("JAVA_SYMBOL", "EntityType<?>", "ModEntities.{constant}"),
    "entity_translation_key": (
        "TRANSLATION_KEY",
        "EntityType<?>",
        "entity.{mod_id}.{registry_path}",
    ),
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
                raise ValueError(
                    f"TEMPLATE_PORT_DECLARATION: {template_id!r} missing {key}"
                )
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

    from .implementation_identity import ExecutorType

    executor = _job_value(job, "executor_type", "")
    if executor == ExecutorType.PYTHON_GENERATOR or executor == "python_generator":
        from .integrity_dispatcher import execute_generator_job

        return execute_generator_job(
            job,
            context=dict(context or {}),
            router=router,
            port_registry=port_registry,
            base_dir=base_dir,
        )

    template_id = str(_job_value(job, "template_id", ""))
    if not template_id:
        raise ValueError("TEMPLATE_JOB_ID: artifact job has no template_id")
    template = load_template(template_id)
    if "render" not in template:
        raise ValueError(
            f"TEMPLATE_NOT_EXECUTABLE: {template_id!r} has no deterministic render mold"
        )

    context_map = dict(context or {})
    from .resolved_version_context import execution_context
    from .version_template_context import (
        resolved_template_values,
        validate_resolved_template_overrides,
    )

    resolved = execution_context(context_map, job)
    if resolved is not None:
        from .integrity_dispatcher import verify_job_binding

        resolved.admit_template(template)
    if resolved is not None and port_registry is not None:
        port_registry.bind_context(resolved.context_id)
    det_inputs = dict(_job_value(job, "deterministic_inputs", {}) or {})
    if resolved is not None:
        validate_resolved_template_overrides(resolved, context_map, det_inputs)
        values: dict[str, Any] = resolved_template_values(
            {
                **context_map,
                **det_inputs,
                "resolved_version_context": resolved,
            }
        )
    else:
        values = {**context_map, **det_inputs}
    if resolved is not None:
        authority = verify_job_binding(job, resolved, context_map)
        from .integrity_dispatcher import canonical_contract

        canonical_contract(job, resolved, context_map, authority)

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
            raise ValueError(
                f"TEMPLATE_AI_SLOT_INVALID: {template_id!r} has a non-object slot"
            )
        slot_id = slot.get("id") or slot.get("slot_id")
        if not slot_id:
            raise ValueError(
                f"TEMPLATE_AI_SLOT_ID: {template_id!r} has an unnamed slot"
            )
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
    if resolved is not None:
        from .integrity_dispatcher import validate_canonical_output

        validation_receipts.extend(
            validate_canonical_output(
                job,
                rendered_output,
                resolved=resolved,
                context=context_map,
                authority=authority,
                target_path=target_file,
                anchor=anchor,
            )
        )
        validation_receipts.append(resolved.validate_artifact(template, rendered_output))
    supported_validators = {
        "java_parse",
        "registry_identifier_unique",
        "registry_identifier",
        "json_parse",
        "json_schema",
        "resource_references",
        "semantic_contract",
        "mod_integration_test",
        "client_side_only",
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
        elif validator_name == "resource_references":
            from .artifact_validators.resource_links import validate_resource_links

            validation_receipts.append(
                validate_resource_links(rendered_output, values, port_registry)
            )
        elif validator_name == "json_parse":
            validation_receipts.append(validate_json_resource(rendered_output))
        elif validator_name == "json_schema":
            from .integrity_validators import validate_json_schema

            schema = template.get("output_schema") or values.get("output_schema")
            if schema is None and resolved is not None:
                schema = resolved.require_fact("schemas", template_id)
            if schema is None:
                raise ValueError("JSON_SCHEMA_REQUIRED")
            validation_receipts.append(
                validate_json_schema(json.loads(rendered_output), schema)
            )
        elif validator_name == "semantic_contract":
            from .integrity_validators import validate_semantic_contract

            validation_receipts.append(
                validate_semantic_contract(
                    rendered_output,
                    contract=values["semantic_contract"],
                    context_id=(
                        resolved.context_id
                        if resolved
                        else values["context_id"]
                    ),
                    leaf_id=_job_value(job, "canonical_leaf", ""),
                )
            )
        elif validator_name == "client_side_only":
            from .integrity_validators import validate_side

            validation_receipts.append(
                validate_side(
                    rendered_output,
                    leaf_id=_job_value(job, "canonical_leaf", ""),
                    side=values["side"],
                    classpath=values["java_classpath"],
                    java_version=str(values["java_version"]),
                )
            )
        elif validator_name == "mod_integration_test":
            from .evidence_store import EvidenceStore
            from .integrity_validators import validate_mod_integration

            validation_receipts.append(
                validate_mod_integration(
                    store=EvidenceStore(Path(values["evidence_store"])),
                    evidence_id=values["evidence_id"],
                    expected=values["evidence_expected"],
                )
            )

    canonical_leaf = str(_job_value(job, "canonical_leaf", "") or "")
    impl_id = str(_job_value(job, "implementation_id", "") or "")
    exec_type = str(_job_value(job, "executor_type", "") or "")
    if canonical_leaf or impl_id or exec_type:
        for receipt in validation_receipts:
            if isinstance(receipt, dict):
                if canonical_leaf and "canonical_leaf" not in receipt:
                    receipt["canonical_leaf"] = canonical_leaf
                if impl_id and "implementation_id" not in receipt:
                    receipt["implementation_id"] = impl_id
                if exec_type and "executor_type" not in receipt:
                    receipt["executor_type"] = exec_type

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
        port["name"] if isinstance(port, Mapping) else port
        for port in logical_outputs
    )

    ports_published: dict[str, Any] = {}
    for logical_name, published_name in zip(
        logical_outputs,
        published_names,
        strict=True,
    ):
        port_obj = _logical_port(logical_name, published_name, values, template_id)
        if resolved is not None:
            from dataclasses import replace

            port_obj = replace(port_obj, context_id=resolved.context_id)
        if published_name in ports_published:
            raise ValueError(f"TEMPLATE_PORT_DUPLICATE: {published_name}")
        if port_registry is not None:
            existing_port = port_registry.get(published_name)
            if existing_port is not None and existing_port != port_obj:
                raise ValueError(f"TEMPLATE_PORT_CONFLICT: {published_name}")
        ports_published[published_name] = port_obj

    effective_base_dir = (
        base_dir
        or context_map.get("project_root")
        or context_map.get("base_dir")
    )
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
                operation=target_spec.get(
                    "operation",
                    _job_value(job, "operation", ""),
                ),
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
            materialize_job,
            rendered_output,
            base_dir=effective_base_dir,
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
        "context_id": resolved.context_id if resolved is not None else "",
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


__all__ = [
    "_STANDARD_PORT_DEFINITIONS",
    "_bind_job_dependencies",
    "_job_value",
    "_logical_port",
    "execute_artifact_template",
]
