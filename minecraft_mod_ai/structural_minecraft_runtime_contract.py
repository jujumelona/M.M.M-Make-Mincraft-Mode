from __future__ import annotations

"""Compile Minecraft work only from the fixed structural translation pipeline."""

from collections.abc import Mapping, Sequence
from typing import Any

from .minecraft_template_steps import ROOT_PROVIDE, TemplateStep
from .structural_artifact_mapping import branch_features_for_artifacts
from .task_template_catalog import load_template
from .translation_runtime import translate_requirement

TEMPLATE_CATALOG_SCHEMA = "mmm/structural-minecraft-tasks"
RESEARCH_BASIS = ()


def _capability(requirement: Mapping[str, Any]) -> str:
    raw = requirement.get("capability")
    if not raw:
        missing = requirement.get("missing_provides")
        if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes, bytearray)) and missing:
            raw = missing[0]
    return str(raw or "structural_requirement").strip().casefold().removeprefix("capability:")


def _append(steps, capability, previous, *, name, outcome, anchor_kinds=("symbol", "test"), branch_features=(), template_id=None):
    output = f"{name}:{capability}"
    steps.append(TemplateStep(name=name, template_id=template_id or f"feature/{name}", outcome=outcome, consumes=(previous,), provides=(output,), anchor_kinds=tuple(anchor_kinds), branch_features=tuple(branch_features)))
    return output


def structural_steps_for_requirement(requirement: Mapping[str, Any]) -> tuple[TemplateStep, ...]:
    capability = _capability(requirement)
    translation = translate_requirement(requirement)
    if translation.unresolved_inputs:
        raise ValueError("STRUCTURAL_ARTIFACT_UNRESOLVED: " + ", ".join(translation.unresolved_inputs))
    steps: list[TemplateStep] = []
    previous = ROOT_PROVIDE
    for name, outcome in (
        ("trigger", "Bind only the declared trigger to its verified entry point"),
        ("input", "Validate only the declared input contract"),
        ("state", "Declare only the required state owner, fields, and defaults"),
        ("transition", "Implement only the declared state transition"),
        ("output", "Expose only the declared observable output"),
        ("failure", "Implement only the declared rejection and preserved-state behavior"),
    ):
        previous = _append(steps, capability, previous, name=name, outcome=f"{outcome} for {capability}", template_id="feature/rules" if name == "failure" else f"feature/{name}")

    for artifact in translation.artifact_kinds:
        manifest = load_template(f"minecraft/{artifact}")
        if manifest.get("execution") != "sequence":
            raise ValueError(f"STRUCTURAL_ARTIFACT_TEMPLATE: minecraft/{artifact} must be a sequence")
        identifiers = manifest.get("steps")
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"STRUCTURAL_ARTIFACT_TEMPLATE: minecraft/{artifact} has no responsibilities")
        artifact_branches = tuple(sorted(branch_features_for_artifacts((artifact,))))
        for identifier in identifiers:
            task = load_template(str(identifier))
            rules = task.get("rules")
            rule_text = " ".join(str(rule).strip() for rule in (rules if isinstance(rules, list) else []) if str(rule).strip())
            outcome = str(task.get("task") or "").strip()
            if rule_text:
                outcome = f"{outcome} {rule_text}".strip()
            previous = _append(
                steps,
                capability,
                previous,
                name=str(identifier).replace("/", "_"),
                template_id=str(identifier),
                outcome=f"{outcome} for {capability}",
                anchor_kinds=tuple(task.get("anchor_kinds") or ("symbol", "test")),
                branch_features=artifact_branches,
            )

    previous = _append(steps, capability, previous, name="integration", template_id="integration/feature_connect", outcome=f"Connect only the declared producer and consumer interfaces for {capability}")
    steps.append(TemplateStep(name="runtime_scenario", template_id="validation/runtime_test", outcome=f"Verify the declared observable acceptance scenarios for {capability}", consumes=(previous,), provides=(capability,), anchor_kinds=("test",), branch_features=()))
    return tuple(steps)


def _required_gates(capability, branches, *, semantic_type="gameplay_mechanic", step=None):
    del capability
    features = set(step.branch_features if step is not None else ())
    gates = ["source_static_validation", "target_compile"]
    if "needs_datagen" in features:
        gates.append("generated_resource_validation")
    if "needs_network" in features:
        gates.append("network_protocol_validation")
    if "needs_worldgen" in features:
        gates.append("worldgen_runtime_validation")
    if "needs_mixin" in features or (semantic_type == "software_quality" and branches.get("needs_mixin", {}).get("status") == "ACTIVE"):
        gates.extend(("behavior_equivalence", "performance_regression"))
    return tuple(dict.fromkeys(gates))


def _compile_tasks(gaps, reuse, target, branches, ownership, *, root_provides=None, emit_trace=True):
    from . import evidence_first_planning as planning
    roots = set(root_provides or {ROOT_PROVIDE})
    reuse_by_req = {str(item["requirement_ref"]): item for item in reuse}
    tasks = []
    for gap in gaps:
        requirement_ref = str(gap["requirement_ref"])
        capability = _capability(gap)
        semantic_type = str(gap.get("semantic_type") or "gameplay_mechanic")
        translation = translate_requirement(gap)
        if translation.unresolved_inputs:
            raise planning.EvidencePlanError("STRUCTURAL_ARTIFACT_UNRESOLVED: " + ", ".join(translation.unresolved_inputs))
        required_provide = str(gap["missing_provides"][0])
        decision = reuse_by_req.get(requirement_ref, {})
        steps = structural_steps_for_requirement(gap)
        dependency_refs = tuple(dict.fromkeys(planning._strings(gap.get("depends_on_requirements"))))
        if dependency_refs:
            steps = planning._rewrite_root(steps, prerequisites=tuple(planning._requirement_done(dep) for dep in dependency_refs))
        if planning._active(branches, "needs_loader_leaf"):
            steps = planning._loader_leaf_steps(capability, steps)
        rewritten = []
        for step in steps:
            provides = tuple(required_provide if item == capability else item for item in step.provides)
            if required_provide in provides:
                provides = tuple(dict.fromkeys((*provides, planning._requirement_done(requirement_ref))))
            rewritten.append(TemplateStep(name=step.name, template_id=step.template_id, outcome=step.outcome, consumes=step.consumes, provides=provides, anchor_kinds=step.anchor_kinds, branch_features=step.branch_features))
        steps = tuple(rewritten)
        for index, step in enumerate(steps):
            task_id = planning._stable_id("task", f"{capability}_{step.name}", {"gap": gap["gap_id"], "index": index})
            active_predicates = [branch for branch, value in branches.items() if value.get("status") == "ACTIVE" and planning._step_uses_branch(step, branch)]
            acceptance = [f"{task_id}: all declared provides exist and all owned anchors pass their integrity checks"]
            if required_provide in step.provides:
                acceptance.extend(str(item) for item in gap.get("acceptance", ()) if planning._is_public_acceptance(item))
            task = {
                "task_id": task_id,
                "semantic_outcome": step.outcome,
                "engineering_worksheet": gap.get("engineering_worksheet"),
                "research_reuse_candidates": gap.get("research_reuse_candidates", []),
                "gap_refs": [gap["gap_id"]],
                "requirement_refs": [requirement_ref],
                "target_cell": dict(target.get("coordinates") or {}),
                "owned_anchors": planning._anchors(capability, step, task_id, ownership),
                "reuse_refs": list(dict.fromkeys([*list(decision.get("component_refs") or ()), *list(decision.get("source_refs") or ())])),
                "consumes": list(step.consumes),
                "provides": list(step.provides),
                "depends_on": [],
                "conditional_predicates": active_predicates,
                "required_gates": list(_required_gates(capability, branches, semantic_type=semantic_type, step=step)),
                "acceptance": list(dict.fromkeys(acceptance)),
                "done_predicate": {"operator": "all", "checks": ["owned_anchor_hashes_recorded", "declared_provides_observed", "required_gates_passed"]},
                "impact_probes": ["changed_symbols", "changed_resource_ids_and_references", "dependency_and_source_set_edges", "affected_tests_and_acceptance_bindings"],
                "template_id": "structural_artifact_pipeline",
                "template_catalog_schema": TEMPLATE_CATALOG_SCHEMA,
                "template_features": sorted(translation.branch_features),
                "state": "pending",
                "task_sha256": "",
            }
            task["task_sha256"] = planning._hash_without(task, "task_sha256")
            tasks.append(task)
    return planning._bind_consumes_dependencies(tasks, root_provides=roots, emit_trace=emit_trace)


def _requirement_branch_features(requirement):
    translation = translate_requirement(requirement)
    if translation.unresolved_inputs:
        raise ValueError("STRUCTURAL_ARTIFACT_UNRESOLVED: " + ", ".join(translation.unresolved_inputs))
    return translation.branch_features



__all__ = ["structural_steps_for_requirement"]
