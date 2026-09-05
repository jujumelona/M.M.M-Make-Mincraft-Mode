from __future__ import annotations

"""Deterministic host baseline from one requirement-bound PlanIR task slice."""

from collections.abc import Mapping, Sequence
from typing import Any

from .research_requirement_schema import FACETS, STRUCTURAL_HINTS


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def requirement_task_slice(
    evidence_plan: Mapping[str, Any], requirement_id: str
) -> tuple[Mapping[str, Any], ...]:
    tasks = evidence_plan.get("tasks")
    if not isinstance(tasks, list):
        return ()
    return tuple(
        task
        for task in tasks
        if isinstance(task, Mapping)
        and requirement_id in _strings(task.get("requirement_refs"))
    )


def _task_anchor_kinds(task: Mapping[str, Any]) -> set[str]:
    anchors = task.get("owned_anchors")
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes, bytearray)):
        return set()
    return {
        str(item.get("kind") or "").casefold()
        for item in anchors
        if isinstance(item, Mapping)
    }


def _task_text(task: Mapping[str, Any]) -> str:
    return " ".join(
        [
            str(task.get("task_id") or ""),
            str(task.get("semantic_outcome") or ""),
            " ".join(_strings(task.get("consumes"))),
            " ".join(_strings(task.get("provides"))),
            " ".join(_strings(task.get("required_gates"))),
            " ".join(_strings(task.get("acceptance"))),
            " ".join(sorted(_task_anchor_kinds(task))),
        ]
    ).casefold()


def _facet_tasks(
    tasks: Sequence[Mapping[str, Any]], facet: str
) -> tuple[Mapping[str, Any], ...]:
    hints = STRUCTURAL_HINTS[facet]
    matched: list[Mapping[str, Any]] = []
    for task in tasks:
        text = _task_text(task)
        kinds = _task_anchor_kinds(task)
        explicit = any(hint in text for hint in hints)
        if facet == "interfaces_integration":
            explicit = explicit or bool(_strings(task.get("consumes")))
        elif facet == "registration_data_resources":
            explicit = explicit or bool(kinds & {"registry_id", "resource"})
        elif facet == "verification_testing":
            explicit = explicit or "test" in kinds or bool(_strings(task.get("acceptance")))
        if explicit:
            matched.append(task)
    return tuple(matched)


def host_facet_baseline(
    requirement: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    parent = str(requirement.get("requirement_id") or "")
    result: dict[str, dict[str, Any]] = {}
    for facet in FACETS:
        matched = _facet_tasks(tasks, facet)
        if not matched:
            result[facet] = {
                "facet": facet,
                "disposition": "not_applicable",
                "statement": "",
                "rationale": (
                    "The frozen requirement-bound PlanIR task slice contains "
                    f"no structural obligation for facet {facet}."
                ),
                "evidence_refs": [],
                "acceptance": [],
                "implementation_obligations": [],
            }
            continue

        task_ids = [
            str(task.get("task_id") or "")
            for task in matched
            if str(task.get("task_id") or "")
        ]
        acceptance = list(
            dict.fromkeys(
                check
                for task in matched
                for check in _strings(task.get("acceptance"))
            )
        )[:8]
        obligations = list(
            dict.fromkeys(
                str(task.get("semantic_outcome") or "").strip()
                for task in matched
                if str(task.get("semantic_outcome") or "").strip()
            )
        )[:8]
        result[facet] = {
            "facet": facet,
            "disposition": "already_covered",
            "statement": f"Frozen PlanIR already covers {facet} for {parent}.",
            "rationale": (
                "Host-owned requirement task slice contains explicit coverage in tasks: "
                + ", ".join(task_ids)
            ),
            "evidence_refs": [],
            "acceptance": acceptance,
            "implementation_obligations": obligations,
        }
    return result


def render_task_slice(
    tasks: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for task in tasks:
        rendered.append(
            {
                "task_id": task.get("task_id"),
                "semantic_outcome": task.get("semantic_outcome"),
                "consumes": list(_strings(task.get("consumes"))),
                "provides": list(_strings(task.get("provides"))),
                "required_gates": list(_strings(task.get("required_gates"))),
                "acceptance": list(_strings(task.get("acceptance")))[:6],
                "anchor_kinds": sorted(_task_anchor_kinds(task)),
            }
        )
    return rendered


__all__ = [
    "host_facet_baseline",
    "render_task_slice",
    "requirement_task_slice",
]
