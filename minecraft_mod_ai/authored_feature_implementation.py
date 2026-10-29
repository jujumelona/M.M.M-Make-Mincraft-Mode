"""Turn original feature obligations into startup-bound Java implementation tasks.

Content registries describe objects. Each gameplay feature also needs its real
interaction/glue implementation; a zero-entity feature cannot vanish here.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

from .artifact_expansion import _canonical_candidate_inputs
from .artifact_job import ArtifactJob
from .authored_plan import AuthoredPlan
from .complete_spec import ProductionModule
from .implementation_identity import ExecutorType
from .registered_leaf_binding import require_registered_leaf_binding
from .request_runtime_records import feature_module_id, owner_records
from .task_template_catalog import canonical_specification_input


def _source_ui_records(plan, requirement_id):
    from .source_ui_actions import deferred_ui_interactions
    return [row for row in deferred_ui_interactions(plan.structured_sections)
            if row["source_requirement_id"] == requirement_id]


def _content_owns_interface(plan, feature, content_modules):
    return (feature.get("implementation_role") == "interface"
            and not _source_ui_records(plan, feature["requirement_id"])
            and any(module.kind == "gui" and feature["requirement_id"] in module.config.get("requirement_refs", ())
                    for module in content_modules))


def _feature_needs_source(plan, feature, context_only):
    return bool(_source_ui_records(plan, feature["requirement_id"])) or not context_only(feature)


def _feature_runtime_bindings(plan, bindings, rid, package_name):
    from copy import deepcopy
    from .source_ui_actions import source_ui_callback
    owners = plan.content_design.get("_request_runtime_owners", ())
    callbacks = [{**source_ui_callback(row, package=package_name), "source_ui_record": deepcopy(row)}
                 for row in _source_ui_records(plan, rid)]
    bindings["owned_runtime_records_json"] = json.dumps(owner_records(plan.structured_sections, owners, rid),
        ensure_ascii=False, separators=(",", ":"))
    bindings["shared_state_contract_json"] = json.dumps(plan.structured_sections.get("state_model", {}),
        ensure_ascii=False, separators=(",", ":"))
    bindings["runtime_state_bindings_json"] = json.dumps({key: value for key, value in
        plan.request_scope.get("behavior_state_bindings", {}).items() if value.get("binding") == "world_state"},
        ensure_ascii=False, separators=(",", ":"))
    bindings["retained_typed_behavior_json"] = json.dumps({
        "event_bindings": plan.typed_plan_ir.get("event_bindings", ()),
        "functions": [{"id": fn["id"], "covers": fn.get("covers", ())} for fn in plan.typed_plan_ir.get("functions", ())],
        "ui_owned_records": owner_records(plan.structured_sections, owners, "__typed_ui__"),
        **({"source_ui_actions": [{"action_id": row["action_id"], "trigger": row["trigger"],
                                   "implementation_status": "source_body_required"} for row in callbacks]} if callbacks else {}),
    }, ensure_ascii=False, separators=(",", ":"))
    bindings["source_ui_callbacks_json"] = json.dumps(callbacks, ensure_ascii=False, separators=(",", ":"))
    if callbacks:
        bindings["source_ui_package_name"] = package_name
        paths = json.loads(bindings["content_dependency_paths_json"])
        source_path = "src/main/java/" + package_name.replace(".", "/") + "/AuthoredSourceActions.java"
        paths.append(source_path)
        bindings["content_dependency_paths_json"] = json.dumps(list(dict.fromkeys(paths)), separators=(",", ":"))
        required = json.loads(bindings["required_content_source_paths_json"])
        bindings["required_content_source_paths_json"] = json.dumps(list(dict.fromkeys([*required, source_path])), separators=(",", ":"))


def _grounded_independent_action(row):
    return (row.get("source_status", "grounded") == "grounded"
            and row.get("trigger") and row.get("effects"))


def _can_isolate_actions(plan, rid, rows):
    """Independence requires disjoint source events and no shared state owner."""
    if len(rows) < 2:
        return False
    if _source_ui_records(plan, rid):
        return False
    if owner_records(plan.structured_sections,
                     plan.content_design.get("_request_runtime_owners", ()), rid):
        return False
    if any((plan.typed_plan_ir or {}).get(key) for key in ("functions", "event_bindings")):
        return False
    if any(row.get(key) for row in rows for key in (
        "state_reads", "state_writes", "runtime_state_reads", "runtime_state_writes"
    )):
        return False
    # Overlapping source occurrences may be one atomic transaction, not two
    # independent jobs. Never split the same trigger into separate writers.
    if len({row.get("source_quote") for row in rows}) != len(rows):
        return False
    if len({row.get("trigger") for row in rows}) != len(rows):
        return False
    return all(_grounded_independent_action(row) for row in rows)


def _independent_feature_work_items(plan, features, behaviors):
    """Emit one unit per independently grounded action, preserving atomic units."""
    from .spec_identity import canonical_spec_id

    items = []
    for feature in features:
        rid = feature["requirement_id"]
        rows = [row for row in behaviors if row.get("requirement_id") == rid]
        if _can_isolate_actions(plan, rid, rows):
            for row in rows:
                subject = canonical_spec_id(feature_module_id(rid) + "_" + row["behavior_id"])
                items.append((feature, row["behavior_id"], subject))
        else:
            items.append((feature, None, feature_module_id(rid)))
    return items


def _context_only_feature(feature, source_behaviors):
    rid = feature["requirement_id"]
    matches = [row for row in source_behaviors
               if row.get("requirement_id") == rid]
    if not matches:
        return False
    statement = str(feature.get("statement") or "").strip()
    return all(
        row.get("source_status") == "source_preserved_unresolved"
        and not str(row.get("trigger") or "").strip()
        and not (row.get("state_reads") or row.get("state_writes")
                 or row.get("runtime_state_reads")
                 or row.get("runtime_state_writes"))
        and str(row.get("intent") or "").strip() == statement
        and str(row.get("effects") or "").strip() == statement
        for row in matches
    )



def compile_request_feature_jobs(
    plan: AuthoredPlan, *, adapter: Any, mod_id: str, package_name: str,
    content_modules: Sequence[Any], content_jobs: Sequence[ArtifactJob],
    runtime_modules: Sequence[Any] = (),
):
    modules, jobs = [], []
    # An existing Typed GUI implementation owns its interface requirement;
    # constructing an additional server callback would duplicate its effects.
    def _owned_interface(feature):
        return _content_owns_interface(plan, feature, content_modules)

    # A broad request heading is not itself an observable gameplay callback.
    # In the October production trace, an unresolved "space mod" heading was
    # converted into a separate Fabric ModInitializer even though it supplied
    # no occurrence, state read or executable effect. Retain this context in
    # the approved request scope, but never ask a 9B model to invent a hook.
    source_behaviors = plan.request_scope.get("behavior_requirements", ())

    non_interface = tuple(feature for feature in plan.request_scope.get(
        "requirements", ()) if not _owned_interface(feature))
    concrete = tuple(feature for feature in non_interface if _feature_needs_source(plan, feature, lambda item: _context_only_feature(item, source_behaviors)))
    if non_interface and not concrete:
        # A request consisting only of ungrounded headings is not a success.
        raise ValueError("REQUEST_FEATURE_EXECUTABLE_BEHAVIOR_REQUIRED")
    execution_features = concrete
    if not execution_features:
        return (), ()
    from .construction_planner import author_behavior_requirements, plan_construction_tasks
    ids = {feature["requirement_id"] for feature in execution_features}
    scoped = {**plan.request_scope, "requirements": execution_features}
    if "behavior_requirements" in scoped:
        scoped["behavior_requirements"] = [
            row for row in scoped["behavior_requirements"]
            if row.get("requirement_id") in ids
        ]
    # Canonicalize the same source-grounded behavior ledger for both the
    # task graph AND artifact generator bindings. Previously only the task
    # view could be corrected while stale, model-invented triggers survived in
    # construction_behaviors_json and re-entered Fabric callback selection.
    admitted_behaviors = {
        behavior.behavior_id: behavior.to_dict()
        for behavior in author_behavior_requirements(scoped, router=None)
    }
    construction_tasks = plan_construction_tasks(
        scoped, (), {}, router=None)
    if not construction_tasks:
        raise ValueError("CONSTRUCTION_TASK_GRAPH_EMPTY")
    for feature, behavior_identity, subject in _independent_feature_work_items(
        plan, execution_features, tuple(admitted_behaviors.values()),
    ):
        rid = feature["requirement_id"]
        # Runtime behavior may be implemented by a GUI, item, entity or other
        # content owner whose catalog provenance was not mapped to this exact
        # request requirement ID. The content graph is produced *before* these
        # feature integration nodes, so the owning source closure must be a
        # dependency regardless of free-text requirement matching.
        # This is a build-order/data-flow edge, never permission to invent
        # an event from a class name. Exact installed Java method signatures
        # are still inspected before a custom event producer can be created.
        prerequisite_modules = [
            item for item in content_modules
            if item.module_id != subject
        ]
        leaf = "minecraft/event/callback"
        binding = require_registered_leaf_binding(adapter.version_context, leaf)
        implementation = binding["implementation"]
        if implementation["implementation_id"] != "python_generator:" + leaf:
            raise ValueError("REQUEST_FEATURE_GENERATOR_REQUIRED")
        selected_intent = admitted_behaviors[behavior_identity]["intent"] if behavior_identity else feature["statement"]
        module = ProductionModule(subject, "integration", {
            "integration_type": "artifact_graph_owned", "requirement_refs": [rid],
            "implementation_obligations": [selected_intent],
        }, depends_on=tuple(dict.fromkeys(
            m.module_id for m in (*prerequisite_modules, *runtime_modules)
        )))
        module.validate()
        inputs = _canonical_candidate_inputs(
            None, canonical_leaf=leaf, mod_id=mod_id, package_name=package_name,
            package_path=package_name.replace(".", "/"), subject=subject,
            minecraft_version=adapter.minecraft_version, context_id=adapter.version_context.context_id,
            source_requirement=selected_intent, parent_requirement=rid,
        )
        spec = canonical_specification_input(leaf, inputs)
        bindings = spec["bindings"]
        fqcn = bindings["package_name"] + "." + bindings["class_name"]
        bindings["request_feature_entrypoint"] = fqcn
        bindings["parent_requirement_id"] = rid
        behavior_rows = [
            admitted_behaviors[row["behavior_id"]]
            for row in plan.request_scope.get("behavior_requirements", ())
            if row.get("requirement_id") == rid
            and (behavior_identity is None or row["behavior_id"] == behavior_identity)
        ]
        if not behavior_rows:
            raise ValueError("CONSTRUCTION_BEHAVIOR_COVERAGE_MISSING: " + rid)
        bindings["construction_behaviors_json"] = json.dumps(
            behavior_rows, ensure_ascii=False, separators=(",", ":"))
        required_task_ids = {row["behavior_id"] + ".logic" for row in behavior_rows}
        current_tasks = [task for task in construction_tasks
                         if rid in task.requirement_ids
                         and (behavior_identity is None or task.behavior_id == behavior_identity)]
        if not required_task_ids <= {task.task_id for task in current_tasks}:
            raise ValueError("CONSTRUCTION_TASK_GRAPH_MISSING_BEHAVIORS: " + rid)
        bindings["construction_tasks_json"] = json.dumps([
            {"task_id": task.task_id, "kind": task.kind,
             "behavior_id": task.behavior_id,
             "intent": task.intent, "depends_on": list(task.depends_on),
             "source_refs": [{
                 "evidence_id": row.get("evidence_id"), "source_id": row.get("source_id"),
                 "status": row.get("status"), "mode": row.get("mode"),
                 "capability": row.get("capability"),
             } for row in task.source_refs],
             "verification_ids": list(task.verification_ids)}
            for task in current_tasks], ensure_ascii=False, separators=(",", ":"))
        bindings["target_contract_json"] = json.dumps(adapter.version_context.to_dict().get("target", {}), separators=(",", ":"))
        declarations = []
        source_paths = []
        java_producer_paths = []
        producer_ports = []
        if runtime_modules:
            source_paths.extend(f"src/main/java/{package_name.replace('.', '/')}/{name}.java"
                                for name in ("AuthoredProgram", "AuthoredStateModel", "AuthoredHostCapabilities"))
        for job in content_jobs:
            if job.owner_module not in module.depends_on:
                continue
            target_path = getattr(job, "target_path", "")
            if target_path.endswith(".java"):
                source_paths.append(target_path)
                java_producer_paths.append(target_path)
                producer_ports.extend(getattr(job, "produces", ()) or ())
                relative = target_path.split("/java/", 1)[-1]
                path = PurePosixPath(relative)
                declaration = {"package_name": str(path.parent).replace("/", "."), "class_name": path.stem}
                declaration.update({key: job.deterministic_inputs[key] for key in (
                    "java_constant", "registry_path",
                ) if key in job.deterministic_inputs})
                declarations.append(declaration)
            canonical_inputs = job.deterministic_inputs.get("_canonical_inputs")
            if canonical_inputs is not None:
                specification = canonical_specification_input(job.canonical_leaf, canonical_inputs)
                if specification.get("language") == "java":
                    source_paths.append(specification["target_path"])
                    java_producer_paths.append(specification["target_path"])
                    producer_ports.extend(getattr(job, "produces", ()) or ())
                    known = specification.get("bindings", {})
                    declarations.append({key: known[key] for key in (
                        "package_name", "class_name", "registry_path", "fact_type",
                    ) if key in known})
        # Concrete producer paths include generator canonical inputs, whose
        # ArtifactJob.target_path is often intentionally empty.
        bindings["required_content_source_paths_json"] = json.dumps(
            list(dict.fromkeys(java_producer_paths)), separators=(",", ":"),
        )
        artifact_owners = {job.owner_module for job in content_jobs}
        for owner in prerequisite_modules:
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
        _feature_runtime_bindings(plan, bindings, rid, package_name)
        bindings["request_features_json"] = json.dumps(
            [{"requirement_id": rid, "statement": selected_intent}],
            ensure_ascii=False, separators=(",", ":"))
        source_pattern = spec["output_schema"]["pattern"] + (
            r".*\bimplements\s+(?:net\.fabricmc\.api\.)?ModInitializer\b.*\bonInitialize\s*\("
        )
        spec["output_schema"]["pattern"] = source_pattern
        spec["slots"][0]["schema"]["pattern"] = source_pattern
        implementation_requirements = (
            "Register the relevant real Fabric/Minecraft callbacks from onInitialize "
            "and implement their behavior. Choose hooks matching the original player/world "
            "interaction; do not default to player join or invent an item/command trigger. "
            "Object declarations, GUI labels, chat confirmations and state counters alone "
            "do not implement world travel/combat/trading or other requested behavior. "
            "Use the actual materialized prerequisite declarations and shared state APIs; "
            "never guess methods from class names. Implement the assigned runtime records, "
            "including their atomic mutations and state transitions. Retained Typed "
            "functions and events keep ownership of their existing implementation records. "
            "source_ui_callbacks_json identifies already-wired UI listeners whose effect "
            "bodies still require your Java implementation here; their dispatchers provide "
            "transport only. Implement those listeners from their exact inputs and original "
            "conditions without registering them twice or inventing worksheet mutations. "
            "Preserve existing command/action identities and other implemented transactions. "
            "Use the existing content owner classes; do not duplicate their registrations."
        )
        bindings["java_implementation_requirements"] = (
            bindings.get("java_implementation_requirements", "") + implementation_requirements
        )
        spec["slots"][0]["description"] += (
            " This is the executable implementation of the supplied original feature. "
            "Implement net.fabricmc.api.ModInitializer and public void onInitialize(); "
            + implementation_requirements
        )
        jobs.append(ArtifactJob(
            job_id=subject + ".implementation", template_id="", owner_module=subject,
            executor_type=ExecutorType.PYTHON_GENERATOR, operation="generate",
            # Module depends_on alone is NOT an executable artifact-graph
            # edge. Gate this source producer on the real published Java
            # output ports, so the callback author cannot race the GUI,
            # item or entity code that exposes its occurrence method.
            requires=tuple(dict.fromkeys(producer_ports)),
            deterministic_inputs={"_canonical_inputs": inputs},
            context_id=adapter.version_context.context_id, canonical_leaf=leaf,
            implementation_id=implementation["implementation_id"],
        ))
        modules.append(module)
    return tuple(modules), tuple(jobs)
