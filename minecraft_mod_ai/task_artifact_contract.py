from __future__ import annotations

"""Project host-owned template tasks into artifacts and acceptance views.

Requirement causality, branch activation and task DAG construction belong exclusively to
``evidence_first_planning``.  This contract does not infer dependencies or reinterpret
acceptance prose.  It only enriches already-compiled tasks with artifact metadata and
adds plan-level artifact/design/acceptance projections.
"""

from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any

from . import evidence_first_planning as _planning
from . import target_grounding_contract as _target_contract

_INSTALLED = False


def _normalize_ownership(
    game_design: Mapping[str, Any], ownership: Mapping[str, Any]
) -> dict[str, Any]:
    del game_design
    value = dict(ownership)
    raw_module = str(value.get("module_id") or ":").strip()
    value["gradle_project_path"] = (
        raw_module if raw_module == ":" or raw_module.startswith(":") else ""
    )
    value["module_id"] = _target_contract._logical_module_id(raw_module, {})
    raw_topology = value.get("topology_module_ids")
    if isinstance(raw_topology, list):
        value["topology_module_ids"] = list(
            dict.fromkeys(
                _target_contract._logical_module_id(str(item), {})
                for item in raw_topology
                if str(item).strip()
            )
        )
    return value


def _artifact_obligations(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    task_id = str(task.get("task_id") or "")
    requirement_refs = list(_planning._strings(task.get("requirement_refs")))
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    kind_map = {
        "symbol": "source_code",
        "resource": "data_or_client_resource",
        "registry_id": "registry_entry",
        "test": "verification_artifact",
        "build_config": "build_configuration",
        "loader_module": "loader_module_binding",
    }
    anchors = task.get("owned_anchors")
    for raw in anchors if isinstance(anchors, list) else ():
        if not isinstance(raw, Mapping):
            continue
        anchor_kind = str(raw.get("kind") or "").strip()
        locator = str(raw.get("locator") or "").strip()
        artifact_kind = kind_map.get(anchor_kind, "implementation_artifact")
        key = (artifact_kind, locator)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "artifact_id": _planning._stable_id(
                    "artifact", artifact_kind, {"task": task_id, "locator": locator}
                ),
                "kind": artifact_kind,
                "locator": locator,
                "requirement_refs": requirement_refs,
                "task_ref": task_id,
                "status": "REQUIRED",
                "provenance_role": "implementation_obligation",
            }
        )

    predicates = set(_planning._strings(task.get("conditional_predicates")))
    feature_artifacts = {
        "needs_datagen": (
            "generated_data_resource",
            "datagen output and reference closure",
        ),
        "needs_client_render": (
            "client_visual_or_ui_resource",
            "client model/texture/UI resource contract",
        ),
        "needs_worldgen": (
            "worldgen_data",
            "world-generation configured/placed/binding data",
        ),
        "needs_persistence": (
            "persistence_schema",
            "serialized state schema and reload compatibility contract",
        ),
        "needs_network": (
            "network_protocol",
            "payload/codec/logical-side/server-validation contract",
        ),
    }
    for predicate, (artifact_kind, purpose) in feature_artifacts.items():
        if predicate not in predicates:
            continue
        key = (artifact_kind, purpose)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "artifact_id": _planning._stable_id(
                    "artifact", artifact_kind, {"task": task_id, "purpose": purpose}
                ),
                "kind": artifact_kind,
                "locator": "unresolved:" + artifact_kind,
                "requirement_refs": requirement_refs,
                "task_ref": task_id,
                "status": "REQUIRED_UNRESOLVED",
                "provenance_role": "implementation_obligation",
                "purpose": purpose,
            }
        )
    return result


def _postprocess_tasks(
    tasks: Sequence[Mapping[str, Any]],
    gaps: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Enrich tasks without changing consumes/provides/dependency architecture."""

    gap_by_req = {
        str(gap.get("requirement_ref") or ""): gap
        for gap in gaps
        if isinstance(gap, Mapping)
    }
    result: list[dict[str, Any]] = []
    for raw in tasks:
        task = dict(raw)
        task_id = str(task.get("task_id") or "")
        req_refs = list(_planning._strings(task.get("requirement_refs")))
        req = req_refs[0] if req_refs else ""
        gap = gap_by_req.get(req, {})

        original_acceptance = list(_planning._strings(task.get("acceptance")))
        public = [
            item for item in original_acceptance if _planning._is_public_acceptance(item)
        ]
        internal = [item for item in original_acceptance if item not in public]
        if not internal:
            internal = [
                f"{task_id}: declared provides, owned anchors, hashes and required gates are internally consistent"
            ]
        task["internal_invariants"] = internal
        task["public_acceptance"] = public
        task["acceptance"] = internal
        task["artifact_obligations"] = _artifact_obligations(task)

        existing_artifact_kinds = {
            str(item.get("kind") or "")
            for item in task["artifact_obligations"]
            if isinstance(item, Mapping)
        }
        gap_obligations = gap.get("artifact_obligations")
        for obligation in gap_obligations if isinstance(gap_obligations, list) else ():
            if not isinstance(obligation, Mapping):
                continue
            kind = str(obligation.get("kind") or "").strip()
            if not kind or kind in existing_artifact_kinds:
                continue
            task["artifact_obligations"].append(
                {
                    "artifact_id": _planning._stable_id(
                        "artifact", kind, {"task": task_id, "requirement": req}
                    ),
                    "kind": kind,
                    "locator": f"unresolved:{kind}",
                    "requirement_refs": req_refs,
                    "task_ref": task_id,
                    "status": "REQUIRED_UNRESOLVED",
                    "provenance_role": "implementation_obligation",
                }
            )
            existing_artifact_kinds.add(kind)

        task["implementation_capabilities"] = list(
            _planning._strings(gap.get("implementation_capabilities"))
        )
        task["design_resolution_obligations"] = list(
            _planning._strings(gap.get("design_resolution_obligations"))
        )
        task["semantic_type"] = str(
            gap.get("semantic_type") or "gameplay_mechanic"
        )
        task["unlock_policy"] = dict(gap.get("unlock_policy") or {})
        task["runtime_acceptance"] = list(
            _planning._strings(gap.get("runtime_acceptance"))
        )

        if public:
            gates = list(_planning._strings(task.get("required_gates")))
            if "runtime_gameplay_validation" not in gates:
                gates.append("runtime_gameplay_validation")
            task["required_gates"] = gates
            done = task.get("done_predicate")
            checks = list(
                _planning._strings(
                    done.get("checks") if isinstance(done, Mapping) else ()
                )
            )
            for check in (
                "public_acceptance_observed",
                "runtime_scenario_receipt_recorded",
                "persistence_network_ui_state_observed_where_applicable",
            ):
                if check not in checks:
                    checks.append(check)
            task["done_predicate"] = {"operator": "all", "checks": checks}

        task["impact_domains"] = list(
            dict.fromkeys(
                "source"
                if item.get("kind") == "source_code"
                else "resources"
                if "resource" in str(item.get("kind"))
                or "worldgen" in str(item.get("kind"))
                else "state"
                if item.get("kind") == "persistence_schema"
                else "network"
                if item.get("kind") == "network_protocol"
                else "build"
                if "build" in str(item.get("kind"))
                or "loader" in str(item.get("kind"))
                else "verification"
                if item.get("kind") == "verification_artifact"
                else "registry"
                for item in task["artifact_obligations"]
            )
        )
        task["task_sha256"] = ""
        task["task_sha256"] = _planning._hash_without(task, "task_sha256")
        result.append(task)
    return tuple(result)


def _artifact_plan(
    plan: Mapping[str, Any], game_design: Mapping[str, Any]
) -> dict[str, Any]:
    required: list[dict[str, Any]] = []
    seen: set[str] = set()
    tasks = plan.get("tasks")
    for task in tasks if isinstance(tasks, list) else ():
        if not isinstance(task, Mapping):
            continue
        artifacts = task.get("artifact_obligations")
        for artifact in artifacts if isinstance(artifacts, list) else ():
            if not isinstance(artifact, Mapping):
                continue
            artifact_id = str(artifact.get("artifact_id") or "")
            if not artifact_id or artifact_id in seen:
                continue
            seen.add(artifact_id)
            required.append(dict(artifact))

    supplied_assets = (
        [dict(item) for item in game_design.get("assets", ()) if isinstance(item, Mapping)]
        if isinstance(game_design.get("assets"), list)
        else []
    )
    resource_kinds = {
        "data_or_client_resource",
        "generated_data_resource",
        "client_visual_or_ui_resource",
        "worldgen_data",
        "item_model",
        "block_model",
        "blockstate",
        "entity_model",
        "recipe",
        "loot_table",
        "tag",
        "lang",
        "dimension_data",
    }
    resource_required = [item for item in required if item.get("kind") in resource_kinds]
    return {
        "schema_version": "mmm/artifact-plan-v2",
        "architecture_owner": "minecraft_template_compiler",
        "required_artifacts": required,
        "supplied_asset_briefs": supplied_assets,
        "asset_requirement_status": (
            "REQUIRED_AND_SUPPLIED"
            if resource_required and supplied_assets
            else "REQUIRED_UNRESOLVED" if resource_required else "NOT_REQUIRED_BY_ARCHITECTURE"
        ),
        "zero_asset_justification": (
            "No selected Minecraft template requires a client/data resource artifact."
            if not resource_required and not supplied_assets
            else ""
        ),
    }


def _design_resolution(plan: Mapping[str, Any]) -> dict[str, Any]:
    obligations: list[dict[str, Any]] = []
    seen: set[str] = set()
    tasks = plan.get("tasks")
    for task in tasks if isinstance(tasks, list) else ():
        if not isinstance(task, Mapping):
            continue
        refs = list(_planning._strings(task.get("requirement_refs")))
        for obligation in _planning._strings(task.get("design_resolution_obligations")):
            obligation_id = _planning._stable_id(
                "design_obligation", obligation, {"task": task.get("task_id")}
            )
            if obligation_id in seen:
                continue
            seen.add(obligation_id)
            obligations.append(
                {
                    "obligation_id": obligation_id,
                    "provenance_role": "implementation_obligation",
                    "requirement_refs": refs,
                    "kind": "template_value_resolution",
                    "reason": obligation,
                    "architecture_owner": "host",
                }
            )
    return {
        "schema_version": "mmm/design-resolution-v2",
        "selected_design_alternatives": [],
        "implementation_obligations": obligations,
        "policy": (
            "Minecraft architecture is selected by the host template catalog. Only "
            "user-specific template values remain design-resolution obligations."
        ),
    }


def _acceptance_boundary(plan: Mapping[str, Any]) -> dict[str, Any]:
    public: list[dict[str, Any]] = []
    request = plan.get("request_catalog")
    requirements = request.get("requirements") if isinstance(request, Mapping) else None
    for requirement in requirements if isinstance(requirements, list) else ():
        if not isinstance(requirement, Mapping):
            continue
        public.append(
            {
                "requirement_ref": requirement.get("requirement_id"),
                "capability": requirement.get("capability"),
                "acceptance": list(_planning._strings(requirement.get("acceptance"))),
                "runtime_acceptance": list(
                    _planning._strings(requirement.get("runtime_acceptance"))
                ),
            }
        )
    internal: list[dict[str, Any]] = []
    tasks = plan.get("tasks")
    for task in tasks if isinstance(tasks, list) else ():
        if not isinstance(task, Mapping):
            continue
        internal.append(
            {
                "task_ref": task.get("task_id"),
                "checks": list(_planning._strings(task.get("internal_invariants"))),
            }
        )
    return {
        "schema_version": "mmm/acceptance-boundary-v2",
        "public_acceptance": public,
        "internal_invariants": internal,
    }


def install_task_artifact_contract() -> None:
    """Install artifact projection without adding a second planning authority."""

    global _INSTALLED
    if _INSTALLED:
        return

    current_ownership = _planning._ownership_context
    if not getattr(current_ownership, "_mmm_logical_module_identity", False):

        @wraps(current_ownership)
        def ownership(game_design: Mapping[str, Any]):
            return _normalize_ownership(game_design, current_ownership(game_design))

        ownership._mmm_logical_module_identity = True
        _planning._ownership_context = ownership

    current_tasks = _planning._compile_tasks
    if not getattr(current_tasks, "_mmm_codeplan_task_artifacts", False):

        @wraps(current_tasks)
        def compile_tasks(
            gaps,
            reuse,
            target,
            branches,
            ownership,
            *,
            root_provides=None,
            emit_trace=True,
        ):
            return _postprocess_tasks(
                current_tasks(
                    gaps,
                    reuse,
                    target,
                    branches,
                    ownership,
                    root_provides=root_provides,
                    emit_trace=emit_trace,
                ),
                gaps,
            )

        compile_tasks._mmm_codeplan_task_artifacts = True
        compile_tasks.__wrapped__ = current_tasks
        _planning._compile_tasks = compile_tasks

    current_compile = _planning.compile_evidence_first_plan
    if not getattr(current_compile, "_mmm_artifact_acceptance_boundary", False):

        @wraps(current_compile)
        def compile_plan(
            prompt: str, game_design: Mapping[str, Any], **kwargs: Any
        ):
            plan = dict(current_compile(prompt, game_design, **kwargs))
            plan["artifact_plan"] = _artifact_plan(plan, game_design)
            plan["design_resolution"] = _design_resolution(plan)
            plan["acceptance_boundary"] = _acceptance_boundary(plan)
            plan["plan_sha256"] = ""
            plan["plan_sha256"] = _planning._hash_without(plan, "plan_sha256")
            _planning.validate_evidence_first_plan(plan, prompt=prompt)
            return plan

        compile_plan._mmm_artifact_acceptance_boundary = True
        _planning.compile_evidence_first_plan = compile_plan

    _INSTALLED = True


__all__ = [
    "_acceptance_boundary",
    "_artifact_plan",
    "_design_resolution",
    "_postprocess_tasks",
    "install_task_artifact_contract",
]
