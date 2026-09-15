from __future__ import annotations

"""Single source of truth for evidence-first execution receipts.

A semantic evidence plan is immutable, but production executes a typed overlay that may
add host-owned execution fields and concrete source anchors.  Receipt production and
validation must therefore consume the *same* lowered execution view.  Keeping a second
canonical-task whitelist here previously allowed the producer and validator schemas to
drift and caused the planner to reject its own receipt.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .evidence_execution_contract import execution_handoff, execution_plan
from .evidence_first_handoff import (
    build_evidence_first_handoff,
    validate_evidence_first_handoff,
)
from .evidence_first_planning import EvidencePlanError, validate_evidence_first_plan
from .plan_collect_all_linker import validate_plan_collect_all

RECEIPT_EXTENSION_FIELDS = frozenset(
    {
        "handoff_sha256",
        "execution_overlay_sha256",
        "production_bindings",
        "asset_bindings",
        "request_context",
    }
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _indexed_objects(
    raw: Any,
    *,
    id_field: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, list):
        raise EvidencePlanError(f"{label} must be a list.")
    indexed = {
        str(item.get(id_field) or ""): dict(item)
        for item in raw
        if isinstance(item, Mapping) and str(item.get(id_field) or "")
    }
    if len(indexed) != len(raw):
        raise EvidencePlanError(f"{label} contains an invalid or duplicate {id_field}.")
    return indexed


def _bindings_by_task(
    raw: Any,
    *,
    task_refs: Sequence[str],
    label: str,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, list):
        raise EvidencePlanError(f"{label} must be a list.")
    result: dict[str, list[dict[str, Any]]] = {task_ref: [] for task_ref in task_refs}
    for item in raw:
        if not isinstance(item, Mapping):
            raise EvidencePlanError(f"{label} entry must be an object.")
        task_ref = str(item.get("task_ref") or "")
        if task_ref not in result:
            raise EvidencePlanError(f"{label} references unknown task {task_ref!r}.")
        result[task_ref].append(dict(item))
    return result


def _dependency_graph(
    work_graph: Mapping[str, Any],
    *,
    task_refs: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    dependencies: dict[str, list[str]] = {task_ref: [] for task_ref in task_refs}
    seen: set[tuple[str, str]] = set()
    raw_edges = work_graph.get("edges")
    if not isinstance(raw_edges, list):
        raise EvidencePlanError("Evidence WorkGraph edges must be a list.")
    for raw_edge in raw_edges:
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
            raise EvidencePlanError(f"Evidence WorkGraph contains invalid edge {pair!r}.")
        if pair in seen:
            raise EvidencePlanError(f"Evidence WorkGraph contains duplicate edge {pair!r}.")
        seen.add(pair)
        dependencies[target].append(source)
    return {task_ref: tuple(items) for task_ref, items in dependencies.items()}


def build_execution_receipt_bundle(
    plan: Mapping[str, Any],
    *,
    canonical_handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Lower ``plan`` once and build the exact receipts consumed by production.

    The returned bundle is the only authority for execution-task shape, overlay bindings,
    request context, and task dependencies.  Producers and validators must not reconstruct
    any of these fields independently.
    """

    validate_evidence_first_plan(plan)
    resolved_canonical = (
        dict(canonical_handoff)
        if canonical_handoff is not None
        else build_evidence_first_handoff(plan)
    )
    validate_evidence_first_handoff(resolved_canonical, source_plan=plan)

    plan_sha256 = str(plan.get("plan_sha256") or "")
    if str(resolved_canonical.get("source_plan_sha256") or "") != plan_sha256:
        raise EvidencePlanError("Evidence handoff is not bound to the exact source plan hash.")

    lowered_plan = execution_plan(plan)
    overlay_handoff = execution_handoff(plan, resolved_canonical, lowered_plan)
    validate_plan_collect_all(lowered_plan, overlay_handoff)

    semantic_tasks = _indexed_objects(
        plan.get("tasks"), id_field="task_id", label="Evidence semantic task catalog"
    )
    execution_tasks = _indexed_objects(
        lowered_plan.get("tasks"), id_field="task_id", label="Evidence execution task catalog"
    )
    requirements = _indexed_objects(
        _mapping(plan.get("request_catalog")).get("requirements"),
        id_field="requirement_id",
        label="Evidence requirement catalog",
    )

    work_graph = _mapping(resolved_canonical.get("work_graph"))
    task_refs = _strings(work_graph.get("task_refs"))
    if not task_refs or tuple(execution_tasks) != task_refs or tuple(semantic_tasks) != task_refs:
        raise EvidencePlanError(
            "Evidence semantic tasks, execution tasks, and WorkGraph order must match exactly."
        )
    dependencies = _dependency_graph(work_graph, task_refs=task_refs)

    production_by_task = _bindings_by_task(
        overlay_handoff.get("production_modules"),
        task_refs=task_refs,
        label="Evidence execution production bindings",
    )
    assets_by_task = _bindings_by_task(
        overlay_handoff.get("asset_requests"),
        task_refs=task_refs,
        label="Evidence execution asset bindings",
    )

    handoff_sha256 = str(resolved_canonical.get("handoff_sha256") or "")
    execution_overlay_sha256 = str(overlay_handoff.get("execution_overlay_sha256") or "")
    if not handoff_sha256 or not execution_overlay_sha256:
        raise EvidencePlanError("Evidence execution receipt hashes are incomplete.")

    request_catalog = _mapping(plan.get("request_catalog"))
    receipts: dict[str, dict[str, Any]] = {}
    for task_ref in task_refs:
        task = dict(execution_tasks[task_ref])
        requirement_refs = _strings(task.get("requirement_refs"))
        unknown = [reference for reference in requirement_refs if reference not in requirements]
        if unknown:
            raise EvidencePlanError(
                f"Evidence execution task {task_ref!r} references unknown requirements {unknown!r}."
            )
        collisions = RECEIPT_EXTENSION_FIELDS.intersection(task)
        if collisions:
            raise EvidencePlanError(
                f"Evidence execution task {task_ref!r} illegally owns receipt fields {sorted(collisions)}."
            )
        receipt = {
            **task,
            "handoff_sha256": handoff_sha256,
            "execution_overlay_sha256": execution_overlay_sha256,
            "production_bindings": production_by_task[task_ref],
            "asset_bindings": assets_by_task[task_ref],
            "request_context": {
                "prompt_sha256": request_catalog.get("prompt_sha256"),
                "requirements": [requirements[reference] for reference in requirement_refs],
                "derived_requirements": list(task.get("derived_requirements") or ()),
            },
        }
        receipts[task_ref] = receipt

    return {
        "plan_sha256": plan_sha256,
        "canonical_handoff": resolved_canonical,
        "execution_plan": lowered_plan,
        "execution_handoff": overlay_handoff,
        "task_refs": task_refs,
        "dependencies": dependencies,
        "receipts": receipts,
    }


def validate_task_receipt(
    embedded: Mapping[str, Any],
    *,
    expected_receipt: Mapping[str, Any],
) -> None:
    """Fail closed unless an embedded receipt is byte-structurally exact."""

    task_id = str(expected_receipt.get("task_id") or "")
    if not task_id:
        raise EvidencePlanError("Expected evidence execution receipt has no task_id.")
    actual_keys = set(embedded)
    expected_keys = set(expected_receipt)
    unknown = actual_keys - expected_keys
    missing = expected_keys - actual_keys
    if unknown:
        raise EvidencePlanError(
            f"Evidence task {task_id!r} contains unrecognized receipt fields: {sorted(unknown)}."
        )
    if missing:
        raise EvidencePlanError(
            f"Evidence task {task_id!r} is missing receipt fields: {sorted(missing)}."
        )
    for key, value in expected_receipt.items():
        if embedded.get(key) != value:
            raise EvidencePlanError(
                f"Evidence task {task_id!r} changed or stale-bound host-owned field {key!r}."
            )


__all__ = [
    "RECEIPT_EXTENSION_FIELDS",
    "build_execution_receipt_bundle",
    "validate_task_receipt",
]
