from __future__ import annotations

"""Live-path composition for the evidence-first production contract.

This module deliberately does not implement another planner, target resolver, branch
classifier, or checkpoint store. Those responsibilities stay with their existing
host-owned components. The only responsibilities here are:

* lower the validated EvidenceFirstPlan through evidence_first_handoff before the
  CompleteGameDesignPlanner creates model-fill templates; and
* enrich the existing CompleteProductionOrchestrator / DurableWorkLedger execution
  receipts with incremental ProjectIndex may-impact evidence.

That keeps one owner for planning semantics, one owner for target selection, one
production handoff, and one durable execution ledger.
"""

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

from .evidence_first_execution import impacted_task_ids_for_paths, refresh_project_index
from .evidence_first_handoff import build_evidence_first_handoff
from .evidence_first_planning import validate_evidence_first_plan
from .plan_collect_all_linker import validate_plan_collect_all
from .project_index import ProjectIndex

_INSTALLED = False


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


def _batches_from_handoff(
    plan: Mapping[str, Any],
    *,
    batch_type: type,
) -> tuple[Any, ...]:
    """Lower one validated semantic plan through the canonical production handoff."""

    validate_evidence_first_plan(plan)
    handoff = build_evidence_first_handoff(plan)
    validate_plan_collect_all(plan, handoff)
    plan_sha256 = str(plan.get("plan_sha256") or "")
    if str(handoff.get("source_plan_sha256") or "") != plan_sha256:
        raise ValueError("Evidence handoff is not bound to the exact source plan hash.")

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
    if set(task_refs) != set(tasks) or len(task_refs) != len(tasks):
        raise ValueError("Evidence handoff task set drifted from the validated semantic plan.")

    dependencies: dict[str, list[str]] = {task_ref: [] for task_ref in task_refs}
    seen_edges: set[tuple[str, str]] = set()
    for raw_edge in work_graph.get("edges", ()):
        edge = _mapping(raw_edge)
        source = str(edge.get("from_task_ref") or "")
        target = str(edge.get("to_task_ref") or "")
        pair = (source, target)
        if (
            not source
            or not target
            or source == target
            or source not in dependencies
            or target not in dependencies
        ):
            raise ValueError(f"Evidence handoff contains invalid WorkGraph edge {pair!r}.")
        if pair in seen_edges:
            raise ValueError(f"Evidence handoff contains duplicate WorkGraph edge {pair!r}.")
        seen_edges.add(pair)
        dependencies[target].append(source)

    production_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in handoff.get("production_modules", ()):
        if not isinstance(item, Mapping):
            raise ValueError("Evidence production binding must be an object.")
        task_ref = str(item.get("task_ref") or "")
        if task_ref not in tasks:
            raise ValueError(
                f"Evidence production binding references unknown task {task_ref!r}."
            )
        production_by_task.setdefault(task_ref, []).append(dict(item))

    assets_by_task: dict[str, list[dict[str, Any]]] = {}
    for item in handoff.get("asset_requests", ()):
        if not isinstance(item, Mapping):
            raise ValueError("Evidence asset binding must be an object.")
        task_ref = str(item.get("task_ref") or "")
        if task_ref not in tasks:
            raise ValueError(f"Evidence asset binding references unknown task {task_ref!r}.")
        assets_by_task.setdefault(task_ref, []).append(dict(item))

    batches: list[Any] = []
    handoff_sha256 = str(handoff.get("handoff_sha256") or "")
    for task_ref in task_refs:
        task = dict(tasks[task_ref])
        task["handoff_sha256"] = handoff_sha256
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
                depends_on_batches=tuple(dict.fromkeys(dependencies.get(task_ref, ()))),
                deliverables=_strings(task.get("provides")),
                exports=(task_ref,),
                task_contract=task,
                evidence_plan_sha256=plan_sha256,
                acceptance_tests=(),
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
    """Attach bounded may-impact evidence without becoming a second state owner."""

    refresh = refresh_project_index(previous_index, index_builder=lambda: current_index)
    current_task_id = str(observation.get("task_id") or "")
    completed = {str(item) for item in completed_task_ids}
    if current_task_id:
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
    unexpected = [path for path in refresh.changed_paths if path not in touched]

    enriched = dict(observation)
    enriched["project_index_refresh"] = {
        "previous_sha256": refresh.previous_sha256,
        "current_sha256": refresh.current_sha256,
        "changed_paths": list(refresh.changed_paths),
    }
    enriched["impact_replan_scope"] = impacted
    enriched["unexpected_drift_paths"] = unexpected
    enriched["replan_required"] = bool(impacted)
    if unexpected:
        reason = "project-index drift requires bounded may-impact replanning"
    elif impacted:
        reason = "observed mutation affects bounded downstream semantic tasks"
    else:
        reason = "observed mutation has no incomplete downstream semantic impact"
    enriched["replan_reason"] = reason
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


def _install_handoff_owner() -> None:
    from . import complete_planner

    current = complete_planner._evidence_host_batches
    if getattr(current, "_mmm_canonical_evidence_handoff", False):
        return

    def evidence_host_batches(plan: Mapping[str, Any]) -> tuple[Any, ...]:
        return _batches_from_handoff(plan, batch_type=complete_planner._ProductionBatch)

    evidence_host_batches._mmm_canonical_evidence_handoff = True  # type: ignore[attr-defined]
    complete_planner._evidence_host_batches = evidence_host_batches


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
            context.project_index.update_files(observation.get("touched_paths", ()))
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
        game_design = getattr(approved, "game_design", None) if approved is not None else None
        plan = (
            game_design.get("_evidence_first_plan")
            if isinstance(game_design, Mapping)
            else None
        )
        project_root = bound.arguments.get("project_root")
        if not isinstance(plan, Mapping) or project_root is None:
            return current_generation(self, *args, **kwargs)

        validate_evidence_first_plan(plan)
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
    """Install only the missing live-path composition exactly once."""

    global _INSTALLED
    if _INSTALLED:
        return
    _install_handoff_owner()
    _install_execution_impact()
    _INSTALLED = True


__all__ = ["install"]
