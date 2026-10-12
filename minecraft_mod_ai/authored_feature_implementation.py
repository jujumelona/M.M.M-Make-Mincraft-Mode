"""Turn original feature obligations into startup-bound Java implementation tasks.

Content registries describe objects. Each gameplay feature also needs its real
interaction/glue implementation; a zero-entity feature cannot vanish here.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from .artifact_expansion import _canonical_candidate_inputs
from .artifact_job import ArtifactJob
from .authored_plan import AuthoredPlan
from .authored_source_rag import bind_owner_source_context
from .complete_spec import ProductionModule
from .implementation_identity import ExecutorType
from .registered_leaf_binding import require_registered_leaf_binding
from .request_runtime_records import feature_module_id, owner_records


def compile_request_feature_jobs(
    plan: AuthoredPlan, *, adapter: Any, mod_id: str, package_name: str,
    content_modules: Sequence[Any], content_jobs: Sequence[ArtifactJob],
    source_rag: Any, source_reuse: Mapping[str, Any],
    runtime_modules: Sequence[Any] = (),
):
    modules, jobs = [], []
    for feature in plan.request_scope.get("requirements", ()):
        rid = feature["requirement_id"]
        owners = [module for module in content_modules if rid in module.config.get("requirement_refs", ())]
        if feature.get("implementation_role") == "interface" and any(m.kind == "gui" for m in owners):
            continue
        leaf = "minecraft/event/callback"
        binding = require_registered_leaf_binding(adapter.version_context, leaf)
        implementation = binding["implementation"]
        if implementation["implementation_id"] != "python_generator:" + leaf:
            raise ValueError("REQUEST_FEATURE_GENERATOR_REQUIRED")
        subject = feature_module_id(rid)
        module = ProductionModule(subject, "integration", {
            "integration_type": "artifact_graph_owned", "requirement_refs": [rid],
            "implementation_obligations": [feature["statement"]],
        }, depends_on=tuple(dict.fromkeys(m.module_id for m in (*owners, *runtime_modules))))
        module.validate()
        inputs = _canonical_candidate_inputs(
            None, canonical_leaf=leaf, mod_id=mod_id, package_name=package_name,
            package_path=package_name.replace(".", "/"), subject=subject,
            minecraft_version=adapter.minecraft_version, context_id=adapter.version_context.context_id,
            source_requirement=feature["statement"], parent_requirement=rid,
        )
        spec = inputs["event_callback_input"]
        bindings = spec["bindings"]
        fqcn = bindings["package_name"] + "." + bindings["class_name"]
        bindings["request_feature_entrypoint"] = fqcn
        bindings["target_contract_json"] = json.dumps(adapter.version_context.to_dict().get("target", {}), separators=(",", ":"))
        declarations = []
        source_paths = []
        if runtime_modules:
            source_paths.extend(f"src/main/java/{package_name.replace('.', '/')}/{name}.java"
                                for name in ("AuthoredProgram", "AuthoredStateModel", "AuthoredHostCapabilities"))
        for job in content_jobs:
            if job.owner_module not in module.depends_on:
                continue
            target_path = getattr(job, "target_path", "")
            if target_path.endswith(".java"):
                source_paths.append(target_path)
                relative = target_path.split("/java/", 1)[-1]
                path = PurePosixPath(relative)
                declaration = {"package_name": str(path.parent).replace("/", "."), "class_name": path.stem}
                declaration.update({key: job.deterministic_inputs[key] for key in (
                    "java_constant", "registry_path",
                ) if key in job.deterministic_inputs})
                declarations.append(declaration)
            for value in job.deterministic_inputs.get("_canonical_inputs", {}).values():
                if isinstance(value, Mapping) and value.get("language") == "java":
                    source_paths.append(value["target_path"])
                    known = value.get("bindings", {})
                    declarations.append({key: known[key] for key in (
                        "package_name", "class_name", "registry_path", "fact_type",
                    ) if key in known})
        artifact_owners = {job.owner_module for job in content_jobs}
        for owner in owners:
            if owner.module_id in artifact_owners:
                continue
            from .extended_content_generator import (
                _JAVA_KINDS,
                _java_path,
                _unit_class_name,
            )
            from .platform_backend_contract import SYSTEM_KIND_TO_PACK

            if owner.kind in _JAVA_KINDS:
                source_paths.append(_java_path(package_name, _unit_class_name(owner.module_id)))
            pack = SYSTEM_KIND_TO_PACK.get(owner.kind)
            if pack:
                from .system_pack_generator import system_pack_source_paths

                source_paths.extend(system_pack_source_paths(pack, package_name))
            if owner.kind in {"entity", "mob"}:
                name = "".join(part.capitalize() for part in owner.module_id.split("_")) + "Entity"
                source_paths.append(f"src/main/java/{package_name.replace('.', '/')}/entity/{name}.java")
        bindings["content_owner_classes_json"] = json.dumps(declarations, separators=(",", ":"))
        bindings["content_dependency_paths_json"] = json.dumps(list(dict.fromkeys(source_paths)), separators=(",", ":"))
        runtime_owners = plan.content_design.get("_request_runtime_owners", ())
        bindings["owned_runtime_records_json"] = json.dumps(
            owner_records(plan.structured_sections, runtime_owners, rid), ensure_ascii=False, separators=(",", ":"))
        bindings["shared_state_contract_json"] = json.dumps(
            plan.structured_sections.get("state_model", {}), ensure_ascii=False, separators=(",", ":"))
        bindings["retained_typed_behavior_json"] = json.dumps({
            "event_bindings": plan.typed_plan_ir.get("event_bindings", ()),
            "functions": [{"id": fn["id"], "covers": fn.get("covers", ())}
                          for fn in plan.typed_plan_ir.get("functions", ())],
            "ui_owned_records": owner_records(plan.structured_sections, runtime_owners, "__typed_ui__"),
        }, ensure_ascii=False, separators=(",", ":"))
        bind_owner_source_context(bindings, owner=module, request_scope=plan.request_scope,
                                  reuse_plan=source_reuse, rag=source_rag, requirement=feature["statement"])
        source_pattern = spec["output_schema"]["pattern"] + (
            r".*\bimplements\s+(?:net\.fabricmc\.api\.)?ModInitializer\b.*\bonInitialize\s*\("
        )
        spec["output_schema"]["pattern"] = source_pattern
        spec["slots"][0]["schema"]["pattern"] = source_pattern
        spec["slots"][0]["description"] += (
            " This is the executable implementation of the supplied original feature. "
            "Implement net.fabricmc.api.ModInitializer and public void onInitialize(); "
            "register the relevant real Fabric/Minecraft callbacks from onInitialize "
            "and implement their behavior. Choose hooks matching the original player/world "
            "interaction; do not default to player join or invent an item/command trigger. "
            "Call installed donor code where applicable and implement missing glue. "
            "Object declarations, GUI labels, chat confirmations and state counters alone "
            "do not implement world travel/combat/trading or other requested behavior. "
            "Use the actual materialized prerequisite declarations and shared state APIs; "
            "never guess methods from class names. Implement the assigned runtime records, "
            "including their atomic mutations and state transitions. The retained Typed "
            "functions/events and UI transactions already execute: do not duplicate their "
            "callbacks, purchases, rewards or state effects. Preserve their exact command "
            "and action identities when adding residual world behavior. "
            "Use the existing content owner classes; do not duplicate their registrations."
        )
        jobs.append(ArtifactJob(
            job_id=subject + ".implementation", template_id="", owner_module=subject,
            executor_type=ExecutorType.PYTHON_GENERATOR, operation="generate",
            deterministic_inputs={"_canonical_inputs": inputs},
            context_id=adapter.version_context.context_id, canonical_leaf=leaf,
            implementation_id=implementation["implementation_id"],
        ))
        modules.append(module)
    return tuple(modules), tuple(jobs)
