from __future__ import annotations

"""Compile Minecraft work only from the fixed structural translation pipeline."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .acceptance_contracts import is_public_acceptance as _is_public_acceptance
from .minecraft_template_steps import ROOT_PROVIDE, TemplateStep
from .root_cause_trace import emit_root_cause
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


class EvidencePlanError(ValueError):
    pass


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha(value: Any) -> str:
    encoded = value.encode("utf-8") if isinstance(value, str) else _canonical(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = dict(value)
    payload[field] = ""
    return _sha(payload)


def _slug(value: Any, fallback: str = "item") -> str:
    raw = str(value or "")
    text = re.sub(r"[^a-z0-9_]+", "_", raw.casefold()).strip("_")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = f"{fallback}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10]}"
    if not text[0].isalpha():
        text = f"{fallback}_{text}"
    return text[:36]


def _stable_id(prefix: str, semantic: str, discriminator: Any) -> str:
    digest = _sha({"semantic": semantic, "discriminator": discriminator})[7:17]
    return f"{prefix}_{_slug(semantic)}_{digest}"[:63]


def _class_name(value: str) -> str:
    words = [item for item in re.split(r"[^A-Za-z0-9]+", value) if item]
    result = "".join(item[:1].upper() + item[1:] for item in words) or "SemanticTask"
    if not result[0].isalpha():
        result = "Task" + result
    return result[:96]


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(text for item in values if (text := str(item).strip())))

def _active(branches: Mapping[str, Mapping[str, Any]], name: str) -> bool:
    value = branches.get(name)
    return isinstance(value, Mapping) and value.get("status") == "ACTIVE"

def _requirement_done(requirement_ref: str) -> str:
    return f"requirement_done:{requirement_ref}"

def _rewrite_root(steps: Sequence[TemplateStep], *, prerequisites: Sequence[str]) -> tuple[TemplateStep, ...]:
    return tuple(
        TemplateStep(
            name=step.name, template_id=step.template_id, outcome=step.outcome,
            consumes=tuple(dict.fromkeys(value for item in step.consumes for value in ((ROOT_PROVIDE, *prerequisites) if item == ROOT_PROVIDE else (item,)))),
            provides=step.provides, anchor_kinds=step.anchor_kinds, branch_features=step.branch_features,
        )
        for step in steps
    )

def _loader_leaf_steps(capability: str, steps: Sequence[TemplateStep]) -> tuple[TemplateStep, ...]:
    common = f"common_contract:{capability}"
    rewritten = []
    for step in steps:
        if capability in step.provides:
            rewritten.append(TemplateStep(name=step.name, template_id=step.template_id, outcome=step.outcome, consumes=step.consumes, provides=tuple(common if item == capability else item for item in step.provides), anchor_kinds=step.anchor_kinds, branch_features=step.branch_features))
        else:
            rewritten.append(step)
    return tuple(rewritten)

def _step_uses_branch(step: TemplateStep, branch: str) -> bool:
    return branch in step.branch_features or (
        branch == "needs_loader_leaf" and step.name == "loader_leaf_binding"
    )


def _anchors(
    capability: str,
    step: TemplateStep,
    task_id: str,
    ownership: Mapping[str, Any],
) -> list[dict[str, Any]]:
    base = _slug(capability)
    class_name = _class_name(task_id)
    namespace_path = str(ownership["namespace"]).replace(".", "/")
    locators = {
        "symbol": (
            f"{ownership['source_root']}/{namespace_path}/mmmplan/{class_name}.{ownership['extension']}"
            f"#{class_name}"
        ),
        "resource": f"resource:{ownership['mod_id']}:{base}/{step.name}",
        "registry_id": f"registry:{ownership['mod_id']}:{base}/{step.name}",
        "test": (
            f"{ownership['test_root']}/{namespace_path}/mmmplan/{class_name}Test.{ownership['extension']}"
            f"#{class_name}Test"
        ),
        "build_config": f"module:{ownership['module_id']}:build_config",
    }
    if step.name == "loader_leaf_binding":
        module_ids = list(_strings(ownership.get("topology_module_ids")))
        if len(module_ids) < 2:
            raise EvidencePlanError(
                "Loader-leaf task requires validated multi-module ownership anchors."
            )
        return [
            {
                "kind": "loader_module",
                "locator": f"module:{module_id}:loader_leaf",
                "ownership": "exclusive",
                "status": "host_reserved",
                "module_id": module_id,
                "source_set": "common" if "common" in module_id.casefold() else "loader_leaf",
            }
            for module_id in module_ids
        ]
    return [
        {
            "kind": kind,
            "locator": locators[kind],
            "ownership": "exclusive",
            "status": "host_reserved",
            "module_id": ownership["module_id"],
            "source_set": (
                "test"
                if kind == "test"
                else "resources"
                if kind == "resource"
                else ownership["source_set"]
            ),
        }
        for kind in step.anchor_kinds
    ]


def _bind_consumes_dependencies(
    tasks: Sequence[Mapping[str, Any]],
    *,
    root_provides: set[str],
    emit_trace: bool = True,
) -> tuple[dict[str, Any], ...]:
    providers: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        for provided in _strings(task.get("provides")):
            providers.setdefault(provided, []).append(task_id)

    bound: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        task_id = str(task["task_id"])
        dependencies: list[str] = []
        for consumed in _strings(task.get("consumes")):
            if consumed in root_provides:
                continue
            candidates = providers.get(consumed, [])
            if len(candidates) != 1:
                raise EvidencePlanError(
                    f"Task {task_id} consumes {consumed!r} without exactly one provider."
                )
            provider = candidates[0]
            if provider == task_id:
                raise EvidencePlanError(
                    f"Task {task_id} consumes its own provide {consumed!r}."
                )
            if provider not in dependencies:
                dependencies.append(provider)
        task["depends_on"] = dependencies
        task["task_sha256"] = ""
        task["task_sha256"] = _hash_without(task, "task_sha256")
        bound.append(task)
        if emit_trace:
            emit_root_cause(
                "task_dependency_bound",
                stage="planning",
                operation="bind_task_dependencies",
                gate="task_dependency_graph",
                result="PASS",
                details={
                    "task_id": task_id,
                    "consumes": task.get("consumes"),
                    "depends_on": dependencies,
                },
            )
    return tuple(bound)


class _PlanningCompat:
    EvidencePlanError = EvidencePlanError
    _strings = staticmethod(_strings)
    _active = staticmethod(_active)
    _requirement_done = staticmethod(_requirement_done)
    _rewrite_root = staticmethod(_rewrite_root)
    _loader_leaf_steps = staticmethod(_loader_leaf_steps)
    _stable_id = staticmethod(_stable_id)
    _step_uses_branch = staticmethod(_step_uses_branch)
    _is_public_acceptance = staticmethod(_is_public_acceptance)
    _anchors = staticmethod(_anchors)
    _hash_without = staticmethod(_hash_without)
    _bind_consumes_dependencies = staticmethod(_bind_consumes_dependencies)

planning = _PlanningCompat()

def _compile_tasks(gaps, reuse, target, branches, ownership, *, root_provides=None, emit_trace=True):
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
