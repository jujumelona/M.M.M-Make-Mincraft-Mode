from __future__ import annotations

"""Live-path composition for the evidence-first production contract.

This module deliberately does not implement another planner, target resolver, branch
classifier, or checkpoint store. Those responsibilities stay with their existing
host-owned components. It consumes the single canonical execution-receipt bundle before
model-fill templates are created. Execution observations are enriched with bounded
ProjectIndex impact evidence.
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

from .evidence_execution_contract import execution_plan
from .evidence_first_execution import impacted_task_ids_for_paths, refresh_project_index
from .evidence_first_planning import validate_evidence_first_plan
from .evidence_task_receipt_contract import build_execution_receipt_bundle
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
    """Lower one validated semantic plan from the canonical execution receipt bundle."""

    bundle = build_execution_receipt_bundle(plan)
    plan_sha256 = str(bundle.get("plan_sha256") or "")
    task_refs = _strings(bundle.get("task_refs"))
    dependencies = _mapping(bundle.get("dependencies"))
    receipts = _mapping(bundle.get("receipts"))
    if not plan_sha256 or not task_refs or tuple(receipts) != task_refs:
        raise ValueError("Evidence execution receipt bundle is incomplete or out of order.")

    batches: list[Any] = []
    for task_ref in task_refs:
        raw_receipt = receipts.get(task_ref)
        if not isinstance(raw_receipt, Mapping):
            raise ValueError(f"Evidence execution receipt is missing task {task_ref!r}.")
        task = dict(raw_receipt)
        dependency_refs = _strings(dependencies.get(task_ref))
        batches.append(
            batch_type(
                batch_id=task_ref,
                scope=str(task.get("semantic_outcome") or ""),
                depends_on_batches=dependency_refs,
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
        lowered = execution_plan(plan)
        tasks = tuple(
            dict(item)
            for item in lowered.get("tasks", ())
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
