from __future__ import annotations

"""Single live-path owner for the evidence-first production contract.

The evidence planner, production handoff, target optimizer and durable work ledger
already exist independently.  This module connects those components once, late in
runtime finalization, without introducing a second planner or checkpoint store.

Live invariants:
* Minecraft conditional branches are derived from structured semantic evidence.
* New/migration targets may not silently fall back from joint reuse-aware selection.
* The validated EvidenceFirstPlan is lowered through evidence_first_handoff before
  CompleteGameDesignPlanner creates model-fill templates.
* DurableWorkLedger remains the only checkpoint owner; semantic task observations are
  enriched with incremental ProjectIndex impact evidence rather than persisted twice.
"""

import hashlib
import inspect
import json
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from .evidence_first_execution import impacted_task_ids_for_paths, refresh_project_index
from .evidence_first_handoff import build_evidence_first_handoff
from .project_index import ProjectIndex

_INSTALLED = False
_BRANCHES = (
    "needs_registry",
    "needs_datagen",
    "needs_persistence",
    "needs_network",
    "needs_client_render",
    "needs_worldgen",
    "needs_mixin",
    "needs_loader_leaf",
)
_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u3131-\u318e\uac00-\ud7a3]+", re.UNICODE)
_BRANCH_SIGNALS = {
    "needs_registry": frozenset(
        {
            "registry",
            "item",
            "block",
            "block_entity",
            "machine",
            "entity",
            "recipe",
            "effect",
            "enchantment",
            "fluid",
            "biome",
            "dimension",
            "아이템",
            "블록",
            "엔티티",
            "레시피",
            "바이옴",
            "차원",
        }
    ),
    "needs_datagen": frozenset(
        {
            "recipe",
            "loot",
            "tag",
            "model",
            "worldgen",
            "datagen",
            "generated_resource",
            "레시피",
            "루트",
            "태그",
            "모델",
            "월드젠",
            "데이터젠",
        }
    ),
    "needs_persistence": frozenset(
        {
            "persistence",
            "saved",
            "storage",
            "serialize",
            "serialization",
            "codec",
            "world_state",
            "저장",
            "영속",
            "직렬화",
            "코덱",
        }
    ),
    "needs_network": frozenset(
        {
            "network",
            "payload",
            "packet",
            "packets",
            "sync",
            "synced",
            "synchronization",
            "네트워크",
            "패킷",
            "동기화",
        }
    ),
    "needs_client_render": frozenset(
        {
            "gui",
            "ui",
            "screen",
            "render",
            "renderer",
            "texture",
            "model",
            "client",
            "hud",
            "화면",
            "렌더",
            "텍스처",
            "클라이언트",
        }
    ),
    "needs_worldgen": frozenset(
        {
            "worldgen",
            "biome",
            "configured_feature",
            "placed_feature",
            "structure",
            "dimension",
            "월드젠",
            "바이옴",
            "구조물",
            "차원",
        }
    ),
    "needs_mixin": frozenset(
        {
            "mixin",
            "optimization",
            "performance",
            "renderer_patch",
            "injection",
            "믹스인",
            "최적화",
            "성능",
            "인젝션",
        }
    ),
    "needs_loader_leaf": frozenset(
        {
            "loader_leaf",
            "multiloader",
            "multi_loader",
            "fabric",
            "forge",
            "neoforge",
            "멀티로더",
            "패브릭",
            "포지",
            "네오포지",
        }
    ),
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _semantic_atoms(requirement: Mapping[str, Any]) -> frozenset[str]:
    source = _mapping(requirement.get("source_span"))
    text = " ".join(
        (
            str(requirement.get("capability") or ""),
            str(requirement.get("statement") or ""),
            str(source.get("text") or ""),
            *(
                str(item)
                for item in requirement.get("provides", ())
                if isinstance(item, str)
            ),
        )
    ).casefold()
    atoms: set[str] = set()
    for token in _TOKEN.findall(text.replace(".", " ").replace("-", " ")):
        atoms.add(token)
        atoms.update(part for part in token.split("_") if part)
    return frozenset(atoms)


def _semantic_branch_predicates(
    requirements: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Derive branch activation from exact semantic atoms and target topology."""

    atoms_by_requirement = {
        str(item.get("requirement_id") or ""): _semantic_atoms(item)
        for item in requirements
        if str(item.get("requirement_id") or "")
    }
    component_kinds = {
        str(item.get("kind") or "").strip().casefold()
        for item in components
        if str(item.get("kind") or "").strip()
    }
    topology = _mapping(target.get("project_topology"))
    loaders = _strings(topology.get("loaders"))

    result: dict[str, dict[str, Any]] = {}
    for branch in _BRANCHES:
        signals = _BRANCH_SIGNALS[branch]
        refs = [
            requirement_ref
            for requirement_ref, atoms in atoms_by_requirement.items()
            if atoms & signals
        ]
        component_evidence = (
            branch == "needs_datagen" and "generated_resource" in component_kinds
        )
        topology_evidence = branch == "needs_loader_leaf" and len(loaders) > 1
        active = bool(refs or component_evidence or topology_evidence)
        evidence_refs = list(refs)
        if component_evidence:
            evidence_refs.append("component-catalog:generated_resource")
        if topology_evidence:
            evidence_refs.append("target-topology:multiple-loaders")
        result[branch] = {
            "predicate": branch,
            "status": "ACTIVE" if active else "NOT_APPLICABLE",
            "evidence_refs": (
                list(dict.fromkeys(evidence_refs))
                if active
                else ["request-catalog:no-matching-capability"]
            ),
            "reason": (
                "activated by exact requirement/component/topology evidence"
                if active
                else "no exact requirement, component, or topology evidence activates this branch"
            ),
        }
    return result


def _require_evidence_backed_optimization(
    result: Any,
    *,
    automatic_target: bool,
) -> Any:
    """Reject the legacy base-optimizer fallback for automatic new/migration targets."""

    if not automatic_target:
        return result
    reuse_plan = getattr(result, "_mmm_reuse_plan", None)
    if not isinstance(reuse_plan, Mapping):
        raise ValueError(
            "Automatic target selection requires joint evidence-backed reuse optimization; "
            "the base platform optimizer fallback is not an admissible target decision."
        )
    target = reuse_plan.get("target")
    capabilities = reuse_plan.get("capabilities")
    if not isinstance(target, Mapping) or not isinstance(capabilities, list) or not capabilities:
        raise ValueError(
            "Automatic target selection produced an incomplete evidence/reuse receipt."
        )
    return result


def _batches_from_handoff(
    plan: Mapping[str, Any],
    *,
    batch_type: type,
) -> tuple[Any, ...]:
    """Lower the plan through the canonical handoff before model-fill batching."""

    handoff = build_evidence_first_handoff(plan)
    tasks = {
        str(item.get("task_id") or ""): dict(item)
        for item in plan.get("tasks", ())
        if isinstance(item, Mapping) and str(item.get("task_id") or "")
    }
    requirements = {
        str(item.get("requirement_id") or ""): dict(item)
        for item in _mapping(plan.get("request_catalog")).get("requirements", ())
        if isinstance(item, Mapping) and str(item.get("requirement_id") or "")
    }

    work_graph = _mapping(handoff.get("work_graph"))
    task_refs = _strings(work_graph.get("task_refs"))
    dependencies: dict[str, list[str]] = {task_ref: [] for task_ref in task_refs}
    for edge in work_graph.get("edges", ()):
        if not isinstance(edge, Mapping):
            continue
        source = str(edge.get("from_task_ref") or "")
        target = str(edge.get("to_task_ref") or "")
        if source in dependencies and target in dependencies:
            dependencies[target].append(source)

    production_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in handoff.get("production_modules", ()):
        if isinstance(item, Mapping):
            production_by_task.setdefault(str(item.get("task_ref") or ""), []).append(
                dict(item)
            )
    assets_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in handoff.get("asset_requests", ()):
        if isinstance(item, Mapping):
            assets_by_task.setdefault(str(item.get("task_ref") or ""), []).append(
                dict(item)
            )

    batches: list[Any] = []
    for task_ref in task_refs:
        task = tasks.get(task_ref)
        if task is None:
            raise ValueError(f"Evidence handoff references unknown task {task_ref!r}.")
        task["handoff_sha256"] = str(handoff.get("handoff_sha256") or "")
        task["production_bindings"] = production_by_task.get(task_ref, [])
        task["asset_bindings"] = assets_by_task.get(task_ref, [])
        task["request_context"] = {
            "prompt_sha256": _mapping(plan.get("request_catalog")).get("prompt_sha256"),
            "requirements": [
                requirements[reference]
                for reference in _strings(task.get("requirement_refs"))
                if reference in requirements
            ],
        }
        batches.append(
            batch_type(
                batch_id=task_ref,
                scope=str(task.get("semantic_outcome") or ""),
                depends_on_batches=tuple(
                    dict.fromkeys(dependencies.get(task_ref, ()))
                ),
                deliverables=_strings(task.get("provides")),
                exports=(task_ref,),
                task_contract=task,
                evidence_plan_sha256=str(plan.get("plan_sha256") or ""),
                acceptance_tests=_strings(task.get("acceptance")),
            )
        )
    return tuple(batches)


def _project_index_snapshot(index: ProjectIndex) -> dict[str, Any]:
    return {
        "files": [
            {"path": item.path, "sha256": item.sha256}
            for item in index.files
        ]
    }


def _observation_hash(observation: Mapping[str, Any]) -> str:
    payload = dict(observation)
    payload.pop("observation_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _enrich_execution_observation(
    observation: Mapping[str, Any],
    *,
    tasks: Sequence[Mapping[str, Any]],
    previous_index: Mapping[str, Any],
    current_index: Mapping[str, Any],
    completed_task_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach bounded index/impact evidence while leaving DurableWorkLedger as owner."""

    refresh = refresh_project_index(previous_index, index_builder=lambda: current_index)
    current_task_id = str(observation.get("task_id") or "")
    completed = {str(item) for item in completed_task_ids}
    completed.add(current_task_id)
    path_impacted = impacted_task_ids_for_paths(
        tasks,
        refresh.changed_paths,
        completed_task_ids=completed,
    )
    declared_downstream = [
        str(item)
        for item in observation.get("affected_downstream_task_ids", ())
        if str(item) and str(item) not in completed
    ]
    impacted = list(dict.fromkeys([*path_impacted, *declared_downstream]))
    touched = {
        str(item).replace("\\", "/")
        for item in observation.get("touched_paths", ())
        if str(item)
    }
    unexpected = [
        path for path in refresh.changed_paths if path not in touched
    ]

    enriched = dict(observation)
    enriched["project_index_refresh"] = {
        "previous_sha256": refresh.previous_sha256,
        "current_sha256": refresh.current_sha256,
        "changed_paths": list(refresh.changed_paths),
    }
    enriched["impact_replan_scope"] = impacted
    enriched["replan_required"] = bool(unexpected)
    enriched["replan_reason"] = (
        "unexpected project-index drift outside the host-observed mutation receipt"
        if unexpected
        else "host-observed task mutation; existing downstream DAG remains valid"
    )
    enriched["observation_sha256"] = _observation_hash(enriched)
    return enriched


@dataclass
class _ExecutionContext:
    project_index: ProjectIndex
    previous_index: Mapping[str, Any]
    tasks: tuple[Mapping[str, Any], ...]
    completed_task_ids: set[str] = field(default_factory=set)


_EXECUTION_CONTEXT: ContextVar[_ExecutionContext | None] = ContextVar(
    "mmm_evidence_first_execution_context",
    default=None,
)


def _install_semantic_branches() -> None:
    from . import evidence_first_planning

    current = evidence_first_planning._branch_predicates
    if getattr(current, "_mmm_exact_evidence_branches", False):
        return
    _semantic_branch_predicates._mmm_exact_evidence_branches = True  # type: ignore[attr-defined]
    evidence_first_planning._branch_predicates = _semantic_branch_predicates


def _install_handoff_owner() -> None:
    from . import complete_planner

    current = complete_planner._evidence_host_batches
    if getattr(current, "_mmm_canonical_evidence_handoff", False):
        return

    def evidence_host_batches(plan: Mapping[str, Any]) -> tuple[Any, ...]:
        return _batches_from_handoff(
            plan,
            batch_type=complete_planner._ProductionBatch,
        )

    evidence_host_batches._mmm_canonical_evidence_handoff = True  # type: ignore[attr-defined]
    complete_planner._evidence_host_batches = evidence_host_batches


def _install_target_hard_gate() -> None:
    from . import platform_resolver

    current = platform_resolver._optimize
    if getattr(current, "_mmm_evidence_target_hard_gate", False):
        return

    @wraps(current)
    def optimize(
        prompt: str,
        *,
        design: dict[str, Any] | None,
        module_kinds: Sequence[str],
        loader_constraint: str | None = None,
        version_constraint: str | None = None,
        target_research_fn: Any | None = None,
    ):
        result = current(
            prompt,
            design=design,
            module_kinds=module_kinds,
            loader_constraint=loader_constraint,
            version_constraint=version_constraint,
            target_research_fn=target_research_fn,
        )
        # Existing-project preservation bypasses _optimize entirely. Any call that
        # reaches this function is a new build or an explicit migration and must not
        # silently downgrade to the legacy base optimizer.
        return _require_evidence_backed_optimization(
            result,
            automatic_target=True,
        )

    optimize._mmm_evidence_target_hard_gate = True  # type: ignore[attr-defined]
    platform_resolver._optimize = optimize


def _install_execution_impact() -> None:
    from . import complete_orchestrator

    current_observation = complete_orchestrator._semantic_execution_observation
    if not getattr(current_observation, "_mmm_evidence_index_impact", False):

        @wraps(current_observation)
        def semantic_execution_observation(*args: Any, **kwargs: Any):
            observation = current_observation(*args, **kwargs)
            context = _EXECUTION_CONTEXT.get()
            if observation is None or context is None:
                return observation
            touched = observation.get("touched_paths", ())
            context.project_index.update_files(touched)
            current_index = _project_index_snapshot(context.project_index)
            enriched = _enrich_execution_observation(
                observation,
                tasks=context.tasks,
                previous_index=context.previous_index,
                current_index=current_index,
                completed_task_ids=tuple(context.completed_task_ids),
            )
            task_id = str(enriched.get("task_id") or "")
            if task_id:
                context.completed_task_ids.add(task_id)
            context.previous_index = current_index
            return enriched

        semantic_execution_observation._mmm_evidence_index_impact = True  # type: ignore[attr-defined]
        complete_orchestrator._semantic_execution_observation = semantic_execution_observation

    cls = complete_orchestrator.CompleteProductionOrchestrator
    current_generation = cls._execute_generation_work
    if getattr(current_generation, "_mmm_evidence_execution_context", False):
        return

    @wraps(current_generation)
    def execute_generation_work(self: Any, *args: Any, **kwargs: Any):
        try:
            bound = inspect.signature(current_generation).bind(self, *args, **kwargs)
            bound.apply_defaults()
        except (TypeError, ValueError):
            return current_generation(self, *args, **kwargs)

        approved = bound.arguments.get("approved")
        game_design = (
            getattr(approved, "game_design", None)
            if approved is not None
            else None
        )
        plan = (
            game_design.get("_evidence_first_plan")
            if isinstance(game_design, Mapping)
            else None
        )
        project_root = bound.arguments.get("project_root")
        if not isinstance(plan, Mapping) or project_root is None:
            return current_generation(self, *args, **kwargs)

        tasks = tuple(
            dict(item)
            for item in plan.get("tasks", ())
            if isinstance(item, Mapping)
        )
        index = ProjectIndex(Path(project_root), policy=self.policy)
        context = _ExecutionContext(
            project_index=index,
            previous_index=_project_index_snapshot(index),
            tasks=tasks,
        )
        token = _EXECUTION_CONTEXT.set(context)
        try:
            return current_generation(self, *args, **kwargs)
        finally:
            _EXECUTION_CONTEXT.reset(token)

    execute_generation_work._mmm_evidence_execution_context = True  # type: ignore[attr-defined]
    cls._execute_generation_work = execute_generation_work


def install() -> None:
    """Install the evidence-first live-path contract exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_semantic_branches()
    _install_handoff_owner()
    _install_target_hard_gate()
    _install_execution_impact()
    _INSTALLED = True


__all__ = ["install"]
