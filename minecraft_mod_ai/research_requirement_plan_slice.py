from __future__ import annotations

"""Deterministic host baseline from one requirement-bound PlanIR task slice.

A facet is considered covered only when the frozen task structure contains executable
signals for it. Acceptance prose alone never closes persistence, networking, lifecycle,
or integration work; this prevents descriptive words such as "persistence-visible" from
being mistaken for an implementation obligation.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .research_requirement_schema import FACETS


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


def _tokens(*values: Any) -> set[str]:
    text = " ".join(str(value or "") for value in values).casefold()
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.replace("_", " ").replace(".", " "))
        if token
    }


def _structural_tokens(task: Mapping[str, Any]) -> set[str]:
    return _tokens(
        task.get("task_id"),
        task.get("semantic_outcome"),
        " ".join(_strings(task.get("consumes"))),
        " ".join(_strings(task.get("provides"))),
        " ".join(_strings(task.get("required_gates"))),
    )


def _acceptance_tokens(task: Mapping[str, Any]) -> set[str]:
    return _tokens(" ".join(_strings(task.get("acceptance"))))


def _has_any(tokens: set[str], values: set[str]) -> bool:
    return bool(tokens & values)


def _facet_is_structurally_covered(task: Mapping[str, Any], facet: str) -> bool:
    kinds = _task_anchor_kinds(task)
    structural = _structural_tokens(task)
    acceptance = _acceptance_tokens(task)
    owns_source = "symbol" in kinds or "build_config" in kinds or "loader_module" in kinds
    owns_test = "test" in kinds

    if facet == "state_lifecycle":
        return owns_source and _has_any(
            structural,
            {
                "state",
                "lifecycle",
                "transition",
                "tick",
                "update",
                "ownership",
                "damage",
                "transaction",
                "assignment",
                "placement",
                "behavior",
            },
        )

    if facet == "interfaces_integration":
        return owns_source and (
            bool(_strings(task.get("consumes")))
            or _has_any(
                structural,
                {
                    "service",
                    "binding",
                    "integration",
                    "surface",
                    "handler",
                    "menu",
                    "screen",
                    "interaction",
                    "transition",
                },
            )
        )

    if facet == "persistence_reload":
        # A verification sentence that merely mentions persistence is not an
        # implementation. Require a source-owning task with a persistence action.
        return owns_source and _has_any(
            structural,
            {
                "persist",
                "persistent",
                "reload",
                "serialize",
                "serialization",
                "deserialize",
                "codec",
                "nbt",
                "save",
                "load",
            },
        )

    if facet == "server_network_authority":
        return owns_source and _has_any(
            structural,
            {
                "server",
                "network",
                "packet",
                "payload",
                "sync",
                "synchronize",
                "authoritative",
                "authority",
                "multiplayer",
            },
        )

    if facet == "registration_data_resources":
        return bool(kinds & {"registry_id", "resource"}) or _has_any(
            structural,
            {
                "registry",
                "register",
                "resource",
                "recipe",
                "loot",
                "tag",
                "model",
                "language",
                "worldgen",
                "datagen",
            },
        )

    if facet == "failure_edge_cases":
        return (owns_source or owns_test) and _has_any(
            structural | acceptance,
            {
                "reject",
                "rejection",
                "invalid",
                "insufficient",
                "incompatible",
                "missing",
                "failure",
                "error",
                "locked",
                "death",
                "remove",
                "replace",
                "fallback",
            },
        )

    if facet == "verification_testing":
        return owns_test or bool(_strings(task.get("acceptance"))) or _has_any(
            structural,
            {
                "test",
                "verify",
                "verification",
                "validation",
                "regression",
                "gametest",
            },
        )

    return False


def _facet_tasks(
    tasks: Sequence[Mapping[str, Any]], facet: str
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        task
        for task in tasks
        if _facet_is_structurally_covered(task, facet)
    )


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
                    f"no executable structural obligation for facet {facet}."
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
            "statement": f"Frozen PlanIR already plans {facet} for {parent}.",
            "rationale": (
                "Host-owned requirement task slice contains executable structural coverage in tasks: "
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
