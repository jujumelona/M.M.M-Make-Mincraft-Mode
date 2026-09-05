from __future__ import annotations

"""Typed lowering from immutable EvidenceFirstPlan to executable task contracts.

The semantic PlanIR remains reproducible and independently validated.  This layer fixes
execution concerns that must not be encoded as fake user requirements: production vs
resource vs verification ownership, task-local source gates, and evidence-backed derived
implementation obligations handed to the coding agent.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .evidence_first_planning import validate_evidence_first_plan

_PRODUCTION_KINDS = frozenset({"symbol", "registry_id", "build_config", "loader_module"})
_RESOURCE_KINDS = frozenset({"resource"})
_TEST_KIND = "test"
_PRODUCTION_COMPILE_GATES = frozenset({"source_static_validation", "target_compile"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _class_name(value: str) -> str:
    words = [item for item in re.split(r"[^A-Za-z0-9]+", value) if item]
    result = "".join(item[:1].upper() + item[1:] for item in words) or "SemanticTask"
    if not result[0].isalpha():
        result = "Task" + result
    return result[:96]


def _anchors(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = task.get("owned_anchors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _runtime_capability(task: Mapping[str, Any]) -> bool:
    return any(value.startswith("capability:") for value in _strings(task.get("provides")))


def _source_anchor(task_id: str, ownership: Mapping[str, Any]) -> dict[str, Any]:
    source_root = str(ownership.get("source_root") or "src/main/java").rstrip("/")
    namespace = str(ownership.get("namespace") or "generated.generated_mod").strip(".")
    namespace_path = namespace.replace(".", "/")
    extension = str(ownership.get("extension") or "java").lstrip(".")
    class_name = _class_name(task_id)
    locator = f"{source_root}/{namespace_path}/mmmplan/{class_name}.{extension}#{class_name}"
    return {
        "kind": "symbol",
        "locator": locator,
        "ownership": "exclusive",
        "status": "host_reserved",
        "module_id": str(ownership.get("module_id") or ":"),
        "source_set": str(ownership.get("source_set") or "main"),
    }


def _derived_for_task(plan: Mapping[str, Any], task: Mapping[str, Any]) -> list[dict[str, Any]]:
    ledger = _mapping(plan.get("derived_requirement_ledger"))
    refs = set(_strings(task.get("requirement_refs")))
    if not refs:
        return []
    result: list[dict[str, Any]] = []
    for raw in ledger.get("facet_decisions", ()):
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("parent_requirement_ref") or "") not in refs:
            continue
        if str(raw.get("disposition") or "") != "derived":
            continue
        result.append(dict(raw))
    return result


def _execution_task(plan: Mapping[str, Any], raw_task: Mapping[str, Any]) -> dict[str, Any]:
    task = json.loads(_canonical(raw_task))
    task_id = str(task.get("task_id") or "")
    ownership = _mapping(plan.get("ownership_context"))
    anchors = _anchors(task)
    kinds = {str(item.get("kind") or "") for item in anchors}
    has_production = bool(kinds & _PRODUCTION_KINDS)
    has_resource = bool(kinds & _RESOURCE_KINDS)
    has_test = _TEST_KIND in kinds

    # A semantic runtime verification step must not be handed to a small coder as a
    # test-only implementation. Give it a concrete production symbol and keep its
    # GameTest anchor so implementation and verification stay in one bounded task.
    if _runtime_capability(task) and has_test and not has_production and not has_resource:
        anchors.append(_source_anchor(task_id, ownership))
        has_production = True
        task["semantic_outcome"] = (
            "Implement the production behavior, then verify the complete semantic outcome: "
            + str(task.get("semantic_outcome") or task_id)
        )
        task["execution_role"] = "production_with_verification"
    elif has_production:
        task["execution_role"] = "production"
    elif has_resource:
        task["execution_role"] = "resource"
    elif has_test:
        task["execution_role"] = "verification"
    else:
        task["execution_role"] = "invalid"

    # source_static_validation and target_compile are production-owner gates in this
    # task graph. Resource/verification work participates through its downstream
    # production owner; assigning these gates locally falsely demands a Java symbol.
    gates = list(_strings(task.get("required_gates")))
    if not has_production:
        gates = [gate for gate in gates if gate not in _PRODUCTION_COMPILE_GATES]
    task["required_gates"] = gates
    task["owned_anchors"] = anchors

    derived = _derived_for_task(plan, task)
    task["derived_requirements"] = derived
    acceptance = list(_strings(task.get("acceptance")))
    implementation_obligations: list[str] = []
    for item in derived:
        acceptance.extend(_strings(item.get("acceptance")))
        implementation_obligations.extend(_strings(item.get("implementation_obligations")))
    task["acceptance"] = list(dict.fromkeys(acceptance))
    task["implementation_obligations"] = list(dict.fromkeys(implementation_obligations))

    task["task_sha256"] = ""
    task["task_sha256"] = _sha(task)
    return task


def execution_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated semantic plan plus a typed, hash-bound execution task overlay."""

    validate_evidence_first_plan(plan)
    result = json.loads(_canonical(plan))
    raw_tasks = plan.get("tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("evidence plan tasks must be a list")
    result["tasks"] = [
        _execution_task(plan, item)
        for item in raw_tasks
        if isinstance(item, Mapping)
    ]
    result["semantic_plan_sha256"] = str(plan.get("plan_sha256") or "")
    result["execution_overlay_sha256"] = _sha(result["tasks"])
    return result


def _binding_id(task_ref: str, anchor: Mapping[str, Any]) -> str:
    return "execution-production-" + _sha(
        {
            "task_ref": task_ref,
            "module_id": anchor.get("module_id"),
            "source_set": anchor.get("source_set"),
            "locator": anchor.get("locator"),
        }
    )[7:27]


def execution_handoff(
    semantic_plan: Mapping[str, Any],
    canonical_handoff: Mapping[str, Any],
    lowered_plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Overlay execution-only bindings without mutating the canonical handoff receipt."""

    validate_evidence_first_plan(semantic_plan)
    lowered = dict(lowered_plan or execution_plan(semantic_plan))
    result = json.loads(_canonical(canonical_handoff))
    modules = [
        dict(item)
        for item in result.get("production_modules", ())
        if isinstance(item, Mapping)
    ]
    bound = {
        (str(item.get("task_ref") or ""), str(anchor.get("locator") or ""))
        for item in modules
        for anchor in item.get("owned_anchors", ())
        if isinstance(anchor, Mapping)
    }
    decisions = {
        str(item.get("requirement_ref") or ""): item
        for item in semantic_plan.get("reuse_decisions", ())
        if isinstance(item, Mapping)
    }
    for task in lowered.get("tasks", ()):
        if not isinstance(task, Mapping):
            continue
        task_ref = str(task.get("task_id") or "")
        refs = _strings(task.get("requirement_refs"))
        actions = {str(decisions[ref].get("action") or "") for ref in refs if ref in decisions}
        action = next(iter(actions)) if len(actions) == 1 else "fresh"
        reuse_refs = list(_strings(task.get("reuse_refs")))
        for anchor in _anchors(task):
            if str(anchor.get("kind") or "") not in _PRODUCTION_KINDS:
                continue
            key = (task_ref, str(anchor.get("locator") or ""))
            if key in bound:
                continue
            modules.append(
                {
                    "production_module_id": _binding_id(task_ref, anchor),
                    "module_id": str(anchor.get("module_id") or ""),
                    "source_set": str(anchor.get("source_set") or ""),
                    "task_ref": task_ref,
                    "requirement_refs": list(refs),
                    "gap_refs": list(_strings(task.get("gap_refs"))),
                    "reuse_action": action,
                    "reuse_refs": reuse_refs,
                    "owned_anchors": [dict(anchor)],
                    "required_gates": list(_strings(task.get("required_gates"))),
                    "execution_overlay": True,
                }
            )
            bound.add(key)
    result["production_modules"] = modules
    result["canonical_handoff_sha256"] = str(canonical_handoff.get("handoff_sha256") or "")
    result["execution_overlay_sha256"] = _sha(
        {
            "semantic_plan_sha256": semantic_plan.get("plan_sha256"),
            "tasks": lowered.get("tasks"),
            "production_modules": modules,
        }
    )
    return result


def task_batches(plan: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Fallback batch lowering when the canonical handoff installer is not active."""

    lowered = execution_plan(plan)
    return tuple(
        {
            "batch_id": task["task_id"],
            "scope": task["semantic_outcome"],
            "depends_on_batches": list(task.get("depends_on") or ()),
            "deliverables": list(task.get("provides") or ()),
            "exports": [task["task_id"]],
            "task_contract": dict(task),
        }
        for task in lowered["tasks"]
    )


__all__ = ["execution_handoff", "execution_plan", "task_batches"]
