from __future__ import annotations

"""Bind cross-system capability prerequisites into the implementation task DAG.

Semantic interpretation, authored requirement identity, source grounding, repair, and
catalog construction belong exclusively to ``semantic_requirement_authority``.  This
contract deliberately acts later: after implementation gaps exist, it adds only those
ontology prerequisites that are themselves unresolved gaps and then asks the canonical
evidence planner to rebind direct task dependencies.

Keeping this boundary narrow avoids duplicate semantic model calls, public requirement
inflation, and stacked planner monkey-patches while preserving causal implementation
ordering.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from . import evidence_first_planning as _evidence
from .canonical_capability_ontology import atomic_capability_definitions

_INSTALLED = False
_MARKER = "__mmm_cross_system_dependencies__"


def _task_capability(gap: Mapping[str, Any]) -> str:
    return (
        str(gap.get("capability") or "")
        .strip()
        .casefold()
        .removeprefix("capability:")
    )


def _would_create_capability_cycle(
    edges: Mapping[str, Sequence[str]],
    consumer: str,
    provider: str,
) -> bool:
    if consumer == provider:
        return True
    stack = [provider]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == consumer:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(str(item) for item in edges.get(current, ()))
    return False


def _compile_tasks_with_cross_system_dependencies(
    gaps: Sequence[Mapping[str, Any]],
    reuse: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    branches: Mapping[str, Mapping[str, Any]],
    ownership: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Add prerequisite consumes to the first task of each unresolved capability.

    The wrapped canonical compiler still owns task identities, per-capability step
    ordering, anchors, hashes, and ordinary consumes/provides edges.  We only connect
    missing capabilities that the ontology declares as prerequisites.  Already verified
    prerequisites do not create redundant tasks or edges.
    """

    original = _compile_tasks_with_cross_system_dependencies.__wrapped__
    tasks = original(gaps, reuse, target, branches, ownership)
    if not tasks or not gaps:
        return tasks

    definitions = atomic_capability_definitions()
    gap_by_capability: dict[str, Mapping[str, Any]] = {}
    first_task_by_gap: dict[str, str] = {}
    required_provide_by_capability: dict[str, str] = {}

    for gap in gaps:
        capability = _task_capability(gap)
        if not capability or capability in gap_by_capability:
            continue
        gap_by_capability[capability] = gap
        missing = [str(item) for item in gap.get("missing_provides", ()) if str(item)]
        if missing:
            required_provide_by_capability[capability] = missing[0]

    mutable = [dict(task) for task in tasks]
    for task in mutable:
        for gap_ref in task.get("gap_refs", ()):
            first_task_by_gap.setdefault(str(gap_ref), str(task["task_id"]))
    task_by_id = {str(task["task_id"]): task for task in mutable}

    accepted_edges: dict[str, list[str]] = {
        capability: [] for capability in gap_by_capability
    }
    for capability, gap in gap_by_capability.items():
        definition = definitions.get(capability)
        if definition is None:
            continue
        consumer_task_id = first_task_by_gap.get(str(gap.get("gap_id") or ""))
        if not consumer_task_id:
            continue
        consumer_task = task_by_id[consumer_task_id]

        for raw_dependency in definition.default_dependencies:
            dependency = str(raw_dependency).strip().casefold()
            if dependency not in gap_by_capability:
                continue
            if _would_create_capability_cycle(
                accepted_edges,
                capability,
                dependency,
            ):
                continue
            required_provide = required_provide_by_capability.get(dependency)
            if not required_provide:
                continue

            consumes = [
                str(item) for item in consumer_task.get("consumes", ()) if str(item)
            ]
            if required_provide not in consumes:
                consumes.append(required_provide)
                consumer_task["consumes"] = consumes
            accepted_edges.setdefault(capability, []).append(dependency)

    return _evidence._bind_consumes_dependencies(
        mutable,
        root_provides={"target:frozen"},
    )


def install() -> None:
    """Install the one planner mutation owned by this contract."""

    global _INSTALLED
    if _INSTALLED:
        return

    original = _evidence._compile_tasks
    if not getattr(original, _MARKER, False):
        _compile_tasks_with_cross_system_dependencies.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(_compile_tasks_with_cross_system_dependencies, _MARKER, True)
        _evidence._compile_tasks = _compile_tasks_with_cross_system_dependencies
    _INSTALLED = True


__all__ = ["install"]
