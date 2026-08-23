"""Deterministic production handoff for evidence-first semantic plans.

This is a lowering layer, not another planner. It consumes a validated evidence-first
plan, preserves the semantic task DAG verbatim, suppresses verified ``retain`` work,
and binds only ``fresh``/``adapt`` tasks to production or resource ownership anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .evidence_first_planning import EvidencePlanError, validate_evidence_first_plan


SCHEMA = "mmm/evidence-first-handoff-v1"
_PRODUCTION_ANCHOR_KINDS = frozenset({"symbol", "registry_id", "build_config", "loader_module"})
_ASSET_ANCHOR_KIND = "resource"
_ALLOWED_ACTIONS = frozenset({"retain", "adapt", "fresh"})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, payload: Any) -> str:
    return f"{prefix}-{_sha(payload)[:20]}"


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True)
class ProductionModule:
    production_module_id: str
    module_id: str
    source_set: str
    task_ref: str
    requirement_refs: tuple[str, ...]
    gap_refs: tuple[str, ...]
    reuse_action: str
    reuse_refs: tuple[str, ...]
    owned_anchors: tuple[Mapping[str, Any], ...]
    required_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "production_module_id": self.production_module_id,
            "module_id": self.module_id,
            "source_set": self.source_set,
            "task_ref": self.task_ref,
            "requirement_refs": list(self.requirement_refs),
            "gap_refs": list(self.gap_refs),
            "reuse_action": self.reuse_action,
            "reuse_refs": list(self.reuse_refs),
            "owned_anchors": [dict(item) for item in self.owned_anchors],
            "required_gates": list(self.required_gates),
        }


@dataclass(frozen=True, slots=True)
class AssetRequest:
    asset_request_id: str
    task_ref: str
    requirement_refs: tuple[str, ...]
    gap_refs: tuple[str, ...]
    reuse_action: str
    reuse_refs: tuple[str, ...]
    locator: str
    module_id: str
    source_set: str
    required_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_request_id": self.asset_request_id,
            "task_ref": self.task_ref,
            "requirement_refs": list(self.requirement_refs),
            "gap_refs": list(self.gap_refs),
            "reuse_action": self.reuse_action,
            "reuse_refs": list(self.reuse_refs),
            "locator": self.locator,
            "module_id": self.module_id,
            "source_set": self.source_set,
            "required_gates": list(self.required_gates),
        }


@dataclass(frozen=True, slots=True)
class RetainReceipt:
    requirement_ref: str
    capability: str
    component_refs: tuple[str, ...]
    suppressed_task_generation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_ref": self.requirement_ref,
            "capability": self.capability,
            "component_refs": list(self.component_refs),
            "suppressed_task_generation": self.suppressed_task_generation,
        }


@dataclass(frozen=True, slots=True)
class WorkGraph:
    task_refs: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_refs": list(self.task_refs),
            "edges": [
                {"from_task_ref": source, "to_task_ref": target}
                for source, target in self.edges
            ],
        }


def _task_action(
    task: Mapping[str, Any],
    decisions_by_requirement: Mapping[str, Mapping[str, Any]],
) -> tuple[str, tuple[str, ...]]:
    requirement_refs = _strings(task.get("requirement_refs"))
    if not requirement_refs:
        raise EvidencePlanError(f"Task {task.get('task_id')!r} has no exact requirement refs.")
    decisions: list[Mapping[str, Any]] = []
    for requirement_ref in requirement_refs:
        decision = decisions_by_requirement.get(requirement_ref)
        if decision is None:
            raise EvidencePlanError(
                f"Task {task.get('task_id')!r} references requirement {requirement_ref!r} without a reuse decision."
            )
        decisions.append(decision)
    actions = {str(item.get("action") or "") for item in decisions}
    if len(actions) != 1 or not actions <= _ALLOWED_ACTIONS:
        raise EvidencePlanError(
            f"Task {task.get('task_id')!r} does not resolve to one exact reuse action."
        )
    action = next(iter(actions))
    if action == "retain":
        raise EvidencePlanError(
            f"Retained requirement leaked into executable task {task.get('task_id')!r}."
        )
    expected_refs: list[str] = []
    for decision in decisions:
        expected_refs.extend(_strings(decision.get("component_refs")))
        expected_refs.extend(_strings(decision.get("source_refs")))
    expected = tuple(dict.fromkeys(expected_refs))
    actual = _strings(task.get("reuse_refs"))
    if actual != expected:
        raise EvidencePlanError(
            f"Task {task.get('task_id')!r} reuse refs do not exactly match its reuse decision."
        )
    if action == "fresh" and actual:
        raise EvidencePlanError(f"Fresh task {task.get('task_id')!r} unexpectedly carries reuse refs.")
    if action == "adapt" and not actual:
        raise EvidencePlanError(f"Adapt task {task.get('task_id')!r} has no verified adaptation refs.")
    return action, actual


def _validate_gap_binding(
    task: Mapping[str, Any],
    gaps_by_id: Mapping[str, Mapping[str, Any]],
) -> None:
    requirement_refs = set(_strings(task.get("requirement_refs")))
    gap_refs = _strings(task.get("gap_refs"))
    if not gap_refs:
        raise EvidencePlanError(f"Task {task.get('task_id')!r} has no exact gap refs.")
    for gap_ref in gap_refs:
        gap = gaps_by_id.get(gap_ref)
        if gap is None:
            raise EvidencePlanError(f"Task {task.get('task_id')!r} references unknown gap {gap_ref!r}.")
        if str(gap.get("requirement_ref") or "") not in requirement_refs:
            raise EvidencePlanError(
                f"Task {task.get('task_id')!r} gap {gap_ref!r} belongs to another requirement."
            )


def _production_modules_for_task(
    task: Mapping[str, Any],
    *,
    action: str,
    reuse_refs: tuple[str, ...],
) -> tuple[ProductionModule, ...]:
    anchors = [
        dict(anchor)
        for anchor in task.get("owned_anchors", ())
        if isinstance(anchor, Mapping) and str(anchor.get("kind") or "") in _PRODUCTION_ANCHOR_KINDS
    ]
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for anchor in anchors:
        key = (str(anchor.get("module_id") or ""), str(anchor.get("source_set") or ""))
        grouped.setdefault(key, []).append(anchor)
    result: list[ProductionModule] = []
    task_ref = str(task["task_id"])
    for (module_id, source_set), owned in sorted(grouped.items()):
        identity = {"task_ref": task_ref, "module_id": module_id, "source_set": source_set}
        result.append(
            ProductionModule(
                production_module_id=_stable_id("production-module", identity),
                module_id=module_id,
                source_set=source_set,
                task_ref=task_ref,
                requirement_refs=_strings(task.get("requirement_refs")),
                gap_refs=_strings(task.get("gap_refs")),
                reuse_action=action,
                reuse_refs=reuse_refs,
                owned_anchors=tuple(owned),
                required_gates=_strings(task.get("required_gates")),
            )
        )
    return tuple(result)


def _asset_requests_for_task(
    task: Mapping[str, Any],
    *,
    action: str,
    reuse_refs: tuple[str, ...],
) -> tuple[AssetRequest, ...]:
    task_ref = str(task["task_id"])
    result: list[AssetRequest] = []
    for anchor in task.get("owned_anchors", ()):
        if not isinstance(anchor, Mapping) or str(anchor.get("kind") or "") != _ASSET_ANCHOR_KIND:
            continue
        locator = str(anchor.get("locator") or "")
        if not locator:
            raise EvidencePlanError(f"Task {task_ref!r} contains a resource anchor without a locator.")
        identity = {"task_ref": task_ref, "locator": locator}
        result.append(
            AssetRequest(
                asset_request_id=_stable_id("asset-request", identity),
                task_ref=task_ref,
                requirement_refs=_strings(task.get("requirement_refs")),
                gap_refs=_strings(task.get("gap_refs")),
                reuse_action=action,
                reuse_refs=reuse_refs,
                locator=locator,
                module_id=str(anchor.get("module_id") or ""),
                source_set=str(anchor.get("source_set") or ""),
                required_gates=_strings(task.get("required_gates")),
            )
        )
    return tuple(result)


def build_evidence_first_handoff(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Lower a validated evidence-first plan into exact production work bindings."""

    validate_evidence_first_plan(plan)
    decisions = plan.get("reuse_decisions")
    tasks = plan.get("tasks")
    gaps = plan.get("gap_catalog")
    if not isinstance(decisions, list) or not isinstance(tasks, list) or not isinstance(gaps, list):
        raise EvidencePlanError("Evidence-first plan is missing handoff catalogs.")

    decisions_by_requirement: dict[str, Mapping[str, Any]] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise EvidencePlanError("Reuse decision must be an object.")
        requirement_ref = str(decision.get("requirement_ref") or "")
        if not requirement_ref or requirement_ref in decisions_by_requirement:
            raise EvidencePlanError("Reuse decisions must bind each requirement exactly once.")
        decisions_by_requirement[requirement_ref] = decision

    gaps_by_id = {
        str(gap.get("gap_id")): gap
        for gap in gaps
        if isinstance(gap, Mapping) and str(gap.get("gap_id") or "")
    }
    if len(gaps_by_id) != len(gaps):
        raise EvidencePlanError("Gap catalog identifiers are missing or duplicated at handoff.")

    retained_requirements = {
        requirement_ref
        for requirement_ref, decision in decisions_by_requirement.items()
        if decision.get("action") == "retain"
    }
    for gap in gaps:
        if str(gap.get("requirement_ref") or "") in retained_requirements:
            raise EvidencePlanError("Retained requirement leaked into the implementation gap catalog.")

    task_refs: list[str] = []
    task_ids: set[str] = set()
    edges: list[tuple[str, str]] = []
    modules: list[ProductionModule] = []
    assets: list[AssetRequest] = []
    for task in tasks:
        if not isinstance(task, Mapping):
            raise EvidencePlanError("Semantic task must be an object at production handoff.")
        task_ref = str(task.get("task_id") or "")
        if not task_ref or task_ref in task_ids:
            raise EvidencePlanError("Semantic task identifiers are missing or duplicated at handoff.")
        task_ids.add(task_ref)
        task_refs.append(task_ref)
        _validate_gap_binding(task, gaps_by_id)
        action, reuse_refs = _task_action(task, decisions_by_requirement)
        modules.extend(_production_modules_for_task(task, action=action, reuse_refs=reuse_refs))
        assets.extend(_asset_requests_for_task(task, action=action, reuse_refs=reuse_refs))
        for dependency in _strings(task.get("depends_on")):
            edges.append((dependency, task_ref))

    for source, target in edges:
        if source not in task_ids or target not in task_ids or source == target:
            raise EvidencePlanError(f"WorkGraph contains invalid exact edge {source!r} -> {target!r}.")

    retain_receipts = [
        RetainReceipt(
            requirement_ref=requirement_ref,
            capability=str(decision.get("capability") or ""),
            component_refs=_strings(decision.get("component_refs")),
        )
        for requirement_ref, decision in sorted(decisions_by_requirement.items())
        if decision.get("action") == "retain"
    ]
    if any(
        retained in _strings(task.get("requirement_refs"))
        for retained in retained_requirements
        for task in tasks
        if isinstance(task, Mapping)
    ):
        raise EvidencePlanError("Retain suppression failed: an executable task still targets retained work.")

    graph = WorkGraph(task_refs=tuple(task_refs), edges=tuple(edges))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
        "source_plan_sha256": str(plan.get("plan_sha256") or ""),
        "production_modules": [item.to_dict() for item in modules],
        "asset_requests": [item.to_dict() for item in assets],
        "retain_receipts": [item.to_dict() for item in retain_receipts],
        "work_graph": graph.to_dict(),
        "handoff_sha256": "",
    }
    payload["handoff_sha256"] = _sha({key: value for key, value in payload.items() if key != "handoff_sha256"})
    validate_evidence_first_handoff(payload, source_plan=plan)
    return payload


def validate_evidence_first_handoff(
    handoff: Mapping[str, Any],
    *,
    source_plan: Mapping[str, Any] | None = None,
) -> None:
    if handoff.get("schema_version") != SCHEMA:
        raise EvidencePlanError("Unsupported evidence-first handoff schema.")
    expected_hash = _sha({key: value for key, value in handoff.items() if key != "handoff_sha256"})
    if handoff.get("handoff_sha256") != expected_hash:
        raise EvidencePlanError("Evidence-first handoff hash mismatch.")
    graph = _mapping(handoff.get("work_graph"))
    task_refs = _strings(graph.get("task_refs"))
    if len(task_refs) != len(set(task_refs)):
        raise EvidencePlanError("WorkGraph task refs are duplicated.")
    task_ids = set(task_refs)
    raw_edges = graph.get("edges")
    if not isinstance(raw_edges, list):
        raise EvidencePlanError("WorkGraph edges must be a list.")
    edges: set[tuple[str, str]] = set()
    for raw in raw_edges:
        edge = _mapping(raw)
        pair = (str(edge.get("from_task_ref") or ""), str(edge.get("to_task_ref") or ""))
        if not all(pair) or pair[0] not in task_ids or pair[1] not in task_ids or pair[0] == pair[1]:
            raise EvidencePlanError("WorkGraph contains a dangling or self edge.")
        if pair in edges:
            raise EvidencePlanError("WorkGraph contains duplicate edges.")
        edges.add(pair)

    for key, id_key in (("production_modules", "production_module_id"), ("asset_requests", "asset_request_id")):
        values = handoff.get(key)
        if not isinstance(values, list):
            raise EvidencePlanError(f"Handoff {key} must be a list.")
        seen: set[str] = set()
        for value in values:
            record = _mapping(value)
            identifier = str(record.get(id_key) or "")
            task_ref = str(record.get("task_ref") or "")
            if not identifier or identifier in seen or task_ref not in task_ids:
                raise EvidencePlanError(f"Handoff {key} contains an invalid exact task binding.")
            seen.add(identifier)
            if record.get("reuse_action") not in {"adapt", "fresh"}:
                raise EvidencePlanError(f"Handoff {key} contains retained or unknown executable work.")

    receipts = handoff.get("retain_receipts")
    if not isinstance(receipts, list):
        raise EvidencePlanError("Retain receipts must be a list.")
    retained = {str(_mapping(item).get("requirement_ref") or "") for item in receipts}
    if "" in retained:
        raise EvidencePlanError("Retain receipt is missing its exact requirement ref.")
    if any(_mapping(item).get("suppressed_task_generation") is not True for item in receipts):
        raise EvidencePlanError("Retain receipt does not prove work suppression.")

    if source_plan is not None:
        validate_evidence_first_plan(source_plan)
        source_tasks = source_plan.get("tasks") if isinstance(source_plan.get("tasks"), list) else []
        expected_task_refs = tuple(str(item["task_id"]) for item in source_tasks if isinstance(item, Mapping))
        expected_edges = {
            (str(dependency), str(item["task_id"]))
            for item in source_tasks
            if isinstance(item, Mapping)
            for dependency in _strings(item.get("depends_on"))
        }
        if task_refs != expected_task_refs or edges != expected_edges:
            raise EvidencePlanError("WorkGraph is not an exact lowering of the semantic task DAG.")
        if handoff.get("source_plan_sha256") != source_plan.get("plan_sha256"):
            raise EvidencePlanError("Handoff is stale for its source evidence-first plan.")
        expected_retained = {
            str(item.get("requirement_ref") or "")
            for item in source_plan.get("reuse_decisions", ())
            if isinstance(item, Mapping) and item.get("action") == "retain"
        }
        if retained != expected_retained:
            raise EvidencePlanError("Retain receipts do not exactly match verified retain decisions.")


__all__ = [
    "AssetRequest",
    "EvidencePlanError",
    "ProductionModule",
    "RetainReceipt",
    "SCHEMA",
    "WorkGraph",
    "build_evidence_first_handoff",
    "validate_evidence_first_handoff",
]
