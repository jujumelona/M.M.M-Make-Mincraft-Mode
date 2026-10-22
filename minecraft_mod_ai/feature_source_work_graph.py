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
        hook = task("occurrence_hook", owner,
                    status="verified" if owner in verified and explicit else "pending",
                    evidence={"event_binding": explicit or None,
                              "available_inspected_event_ids": available_ids})
        producers = []
        for row in by_behavior.get(owner, ()):
            if row.get("reason") == "occurrence_hook_not_owned":
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


def member_source_work(graph: Mapping[str, Any], behavior_id: str, *, member_role: str) -> dict[str, Any]:
    """Provide just this member's hook/producer/effect ownership chain."""
    rows = [dict(row) for row in graph.get("tasks", ())
            if row.get("behavior_id") == behavior_id]
    role = str(member_role)
    if role == "register_callback":
        active = {"occurrence_hook", "event_registration"}
    elif role in {"implement_callback_effects", "test_callback_condition"}:
        active = {"occurrence_hook", "runtime_producer", "gameplay_effect"}
    else:
        active = {"occurrence_hook", "runtime_producer", "gameplay_effect",
                  "event_registration"}
    return {
        "schema": "mmm/member-source-work-v1",
        "behavior_id": behavior_id,
        "source_tasks": [row for row in rows if row.get("kind") in active],
        "require_real_observer_and_registration": True,
    }
