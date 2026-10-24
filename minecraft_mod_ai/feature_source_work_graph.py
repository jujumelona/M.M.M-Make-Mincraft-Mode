"""Verified dependency ledger for authored gameplay.

An AI-authored state label is never proof of a runtime producer. The compiler
cannot certify behavior whose callback, observation, or effects were never
installed. The graph preserves missing work without substituting a trigger.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def _id(kind: str, owner: str, name: str = "") -> str:
    digest = hashlib.sha256((owner + "\0" + name).encode()).hexdigest()[:16]
    return kind + ":" + digest


def build_source_work_graph(
    behaviors: Sequence[Mapping[str, Any]],
    *,
    producer_obligations: Sequence[Mapping[str, Any]],
    inspected_hooks: Sequence[Mapping[str, Any]],
    verified_behavior_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Order actual work, without inventing a producer or assuming any hook.

    The inspected Fabric event ABI is an available API, not evidence that an
    occurrence matches the original gameplay requirement.
    """
    by_behavior: dict[str, list[dict[str, Any]]] = {}
    for obligation in producer_obligations:
        owner = str(obligation.get("behavior_id") or "")
        if not owner:
            raise ValueError("FEATURE_WORK_OBLIGATION_OWNER_REQUIRED")
        by_behavior.setdefault(owner, []).append(dict(obligation))
    available_ids = sorted({str(row.get("event_id") or "") for row in inspected_hooks
                            if str(row.get("event_id") or "")})
    verified = set(verified_behavior_ids)
    tasks = []
    seen = set()

    def task(kind, owner, *, name="", depends=(), status="pending", evidence=None):
        identity = _id(kind, owner, name)
        if identity in seen:
            raise ValueError("FEATURE_WORK_DUPLICATE_TASK: " + identity)
        seen.add(identity)
        record = {
            "task_id": identity, "kind": kind, "behavior_id": owner,
            "name": name, "depends_on": list(dict.fromkeys(depends)),
            "status": status,
        }
        if evidence is not None:
            record["evidence"] = evidence
        tasks.append(record)
        return identity

    behavior_ids = set()
    for behavior in behaviors:
        owner = str(behavior.get("behavior_id") or "")
        if not owner or owner in behavior_ids:
            raise ValueError("FEATURE_WORK_BEHAVIOR_ID_REQUIRED_OR_DUPLICATE")
        behavior_ids.add(owner)
        explicit = str(behavior.get("event_binding") or "")
        if explicit and explicit not in available_ids:
            raise ValueError("FEATURE_WORK_EXPLICIT_HOOK_NOT_INSPECTED: " + explicit)
        # A missing occurrence source is a *producer job*, not a reason
        # to pick an unrelated callback. It must exist before its listener.
        owner_obligations = by_behavior.get(owner, ())
        missing_occurrence = [row for row in owner_obligations if row.get("reason") in {
            "occurrence_hook_not_owned", "no_inspected_callback",
        }]
        source_dep = ()
        if missing_occurrence:
            source_id = task("custom_event_source", owner, evidence={
                "trigger": str(behavior.get("trigger") or ""),
                "source_quote": str(behavior.get("source_quote") or ""),
                "required_action": "implement an inspected source-site event or explicit typed UI action",
                "reasons": sorted({str(row["reason"]) for row in missing_occurrence}),
            })
            source_dep = (source_id,)
        # An absent observation is a separate data-production task. The
        # occurrence hook must be selected first to establish real typed
        # receivers; then observers may be implemented against those inputs
        # before gameplay effects run. Only an absent occurrence itself
        # requires a separate source owner before hook selection.
        before_hook_reasons = {
            "no_inspected_semantic_source",
            "missing_world_observer_binding",
            "no_single_callback_supplies_required_sources",
        }
        hook = task("occurrence_hook", owner,
                    depends=source_dep,
                    status="verified" if owner in verified and explicit and not source_dep else "pending",
                    evidence={"event_binding": explicit or None,
                              "available_inspected_event_ids": available_ids})
        producers = []
        for row in owner_obligations:
            if row.get("reason") not in before_hook_reasons:
                continue
            name = str(row.get("name") or row.get("reason") or "")
            if not name:
                raise ValueError("FEATURE_WORK_PRODUCER_OBLIGATION_UNNAMED")
            producer = task("runtime_producer", owner, name=name,
                            depends=(hook,), evidence={
                                "reason": row.get("reason"),
                                "required_action": row.get("required_action"),
                                "source_order": "after_occurrence_hook_before_effect",
                            })
            producers.append(producer)
        for row in owner_obligations:
            if row.get("reason") in {
                "occurrence_hook_not_owned", "no_inspected_callback",
                *before_hook_reasons,
            }:
                continue
            name = str(row.get("name") or row.get("reason") or "")
            if not name:
                raise ValueError("FEATURE_WORK_PRODUCER_OBLIGATION_UNNAMED")
            producers.append(task("runtime_producer", owner, name=name,
                                  depends=(hook,), evidence={
                                      "reason": row.get("reason"),
                                      "required_action": row.get("required_action"),
                                  }))
        effect = task("gameplay_effect", owner, depends=(hook, *producers), evidence={
            "intent": str(behavior.get("intent") or ""),
            "effects": str(behavior.get("effects") or ""),
            "source_quote": str(behavior.get("source_quote") or ""),
        })
        task("event_registration", owner, depends=(hook, effect))
    unknown = set(by_behavior) - behavior_ids
    if unknown:
        raise ValueError("FEATURE_WORK_OBLIGATION_UNKNOWN_BEHAVIOR: " + ",".join(sorted(unknown)))
    compile_id = task("target_compile", "__feature__", depends=(
        row["task_id"] for row in tasks if row["kind"] == "event_registration"))
    task("runtime_playtest", "__feature__", depends=(compile_id,))
    positions = {row["task_id"]: i for i, row in enumerate(tasks)}
    for i, row in enumerate(tasks):
        if any(dep not in positions or positions[dep] >= i for dep in row["depends_on"]):
            raise ValueError("FEATURE_WORK_DEPENDENCY_CYCLE_OR_UNBOUND")
    return {
        "schema": "mmm/source-feature-work-v1",
        "tasks": tasks,
        "unverified_obligations": sum(row["status"] != "verified" for row in tasks),
        "completion_verified": False,
    }


def bind_selected_hooks(
    graph: Mapping[str, Any], selected: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Record inspected hook selection, without certifying runtime behavior.

    An inspected Fabric callback was CHOSEN, not yet proven to produce the
    original observation; dependent work must remain pending.
    """
    from copy import deepcopy

    amended = deepcopy(dict(graph))
    registered = set()
    for row in amended.get("tasks", ()):
        if row.get("kind") != "occurrence_hook":
            continue
        owner = row.get("behavior_id")
        if owner not in selected:
            continue
        event = selected[owner]
        selected_id = event.get("event_id")
        if not isinstance(selected_id, str) or not selected_id:
            raise ValueError("FEATURE_WORK_SELECTED_EVENT_ID_REQUIRED")
        offered = set(row.get("evidence", {}).get("available_inspected_event_ids") or ())
        materialized = set(row.get("evidence", {}).get("materialized_custom_event_ids") or ())
        if selected_id not in offered and selected_id not in materialized:
            raise ValueError("FEATURE_WORK_SELECTED_HOOK_NOT_INSPECTED: " + selected_id)
        row["evidence"]["event_binding"] = selected_id
        if selected_id in materialized:
            row["evidence"]["source_kind"] = "materialized_custom_mixin"
        if row.get("status") != "verified":
            row["status"] = "source_selected"
        registered.add(owner)
    if set(selected) - registered:
        raise ValueError("FEATURE_WORK_SELECTED_HOOK_OWNER_UNKNOWN")
    amended["unverified_obligations"] = sum(
        row.get("status") != "verified" for row in amended.get("tasks", ()))
    amended["completion_verified"] = False
    return amended


def member_source_work(graph: Mapping[str, Any], behavior_id: str, *, member_role: str) -> dict[str, Any]:
    """Provide just this member's hook/producer/effect ownership chain."""
    rows = [dict(row) for row in graph.get("tasks", ())
            if row.get("behavior_id") == behavior_id]
    role = str(member_role)
    if role == "register_callback":
        active = {"custom_event_source", "occurrence_hook", "event_registration"}
    elif role in {"implement_callback_effects", "test_callback_condition"}:
        active = {"custom_event_source", "occurrence_hook", "runtime_producer", "gameplay_effect"}
    else:
        active = {"custom_event_source", "occurrence_hook", "runtime_producer",
                  "gameplay_effect", "event_registration"}
    return {
        "schema": "mmm/member-source-work-v1",
        "behavior_id": behavior_id,
        "source_tasks": [row for row in rows if row.get("kind") in active],
        "require_real_observer_and_registration": True,
    }


def requires_original_occurrence_site(graph: Mapping[str, Any], behavior_id: str) -> bool:
    """Do not confuse missing runtime *data* with a missing occurrence *event*.

    A runtime producer derives data (e.g. visible planets, prerequisites)
    from a real receiver after an occurrence hook has been selected. A
    custom Mixin is mandatory only when the occurrence itself has no
    observed source. Otherwise demanding an arbitrary source-site method
    conflates two different dependencies and hides legitimate Fabric event
    receiver types from the author.
    """
    return any(
        task.get("kind") == "custom_event_source"
        and task.get("behavior_id") == behavior_id
        and task.get("status") != "verified"
        for task in graph.get("tasks", ())
    )
