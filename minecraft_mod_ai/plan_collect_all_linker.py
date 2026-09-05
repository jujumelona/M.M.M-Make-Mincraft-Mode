from __future__ import annotations

"""Collect-all deterministic linker between semantic tasks and production execution.

The linker is deliberately model-free. It validates the entire handoff before any coder
request is allowed, reports every deterministic defect in one exception, and treats
semantic locators, test artifacts, and executable production paths as distinct types.
"""

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .root_cause_trace import emit_root_cause

_SOURCE_ROOTS = (
    "src/main/java/",
    "src/client/java/",
)
_TEST_ROOTS = (
    "src/test/",
    "src/gametest/",
)
_COMPILE_GATES = frozenset({"source_static_validation", "target_compile"})


@dataclass(frozen=True, slots=True)
class PlanLinkIssue:
    code: str
    task_ref: str
    message: str
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "task_ref": self.task_ref,
            "message": self.message,
            "details": dict(self.details),
        }


class PlanCollectAllLinkError(RuntimeError):
    def __init__(self, issues: Sequence[PlanLinkIssue]) -> None:
        self.issues = tuple(issues)
        summary = "; ".join(
            f"{item.code}[{item.task_ref or '-'}]: {item.message}"
            for item in self.issues
        )
        super().__init__(
            f"PLAN_COLLECT_ALL_PREFLIGHT_FAILED ({len(self.issues)} issue(s)): {summary}"
        )


def _strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        return ()
    return tuple(str(item).strip() for item in values if str(item).strip())


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _anchors(task: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = task.get("owned_anchors")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in raw if isinstance(item, Mapping))


def _locator(anchor: Mapping[str, Any]) -> str:
    return str(anchor.get("locator") or "").replace("\\", "/").strip()


def _path_from_locator(locator: str) -> str:
    raw = str(locator or "").replace("\\", "/").strip()
    if not raw or raw.startswith("/") or ":" in raw.split("#", 1)[0]:
        return ""
    path = raw.split("#", 1)[0]
    while path.startswith("./"):
        path = path[2:]
    parts = PurePosixPath(path).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return ""
    return PurePosixPath(path).as_posix()


def _is_source_symbol(anchor: Mapping[str, Any]) -> bool:
    if str(anchor.get("kind") or "") != "symbol":
        return False
    path = _path_from_locator(_locator(anchor))
    return bool(path and path.endswith(".java") and path.startswith(_SOURCE_ROOTS))


def _is_test_path(anchor: Mapping[str, Any]) -> bool:
    path = _path_from_locator(_locator(anchor))
    return bool(path and path.startswith(_TEST_ROOTS))


def _claims_runtime(task: Mapping[str, Any]) -> bool:
    provides = _strings(task.get("provides"))
    if any(value.startswith("capability:") for value in provides):
        return True
    semantic = str(task.get("semantic_outcome") or "").casefold()
    if semantic and semantic not in {"test", "verification", "resource", "asset"}:
        return True
    return False


def _cycle_nodes(task_ids: Sequence[str], edges: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    indegree = {task_id: 0 for task_id in task_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        if source in indegree and target in indegree and source != target:
            outgoing[source].append(target)
            indegree[target] += 1
    queue = deque(task_id for task_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in outgoing.get(source, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited == len(indegree):
        return ()
    return tuple(sorted(task_id for task_id, degree in indegree.items() if degree > 0))


def collect_plan_link_issues(
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> tuple[PlanLinkIssue, ...]:
    """Return every deterministic task/binding defect without fail-fast behavior."""

    issues: list[PlanLinkIssue] = []
    raw_tasks = plan.get("tasks")
    tasks = raw_tasks if isinstance(raw_tasks, list) else []

    task_items: list[tuple[str, Mapping[str, Any]]] = []
    seen_task_ids: set[str] = set()
    for index, raw in enumerate(tasks):
        if not isinstance(raw, Mapping):
            issues.append(
                PlanLinkIssue(
                    "TASK_NOT_OBJECT",
                    "",
                    "semantic task must be an object",
                    {"task_index": index},
                )
            )
            continue
        task_id = str(raw.get("task_id") or "").strip()
        if not task_id:
            issues.append(
                PlanLinkIssue(
                    "TASK_ID_MISSING",
                    "",
                    "semantic task has no stable task_id",
                    {"task_index": index},
                )
            )
            continue
        if task_id in seen_task_ids:
            issues.append(
                PlanLinkIssue(
                    "TASK_ID_DUPLICATE",
                    task_id,
                    "task_id is produced more than once",
                    {"task_index": index},
                )
            )
        seen_task_ids.add(task_id)
        task_items.append((task_id, raw))

    known_ids = {task_id for task_id, _task in task_items}
    edges: list[tuple[str, str]] = []
    for task_id, task in task_items:
        for dependency in _strings(task.get("depends_on")):
            if dependency == task_id:
                issues.append(
                    PlanLinkIssue(
                        "TASK_SELF_DEPENDENCY",
                        task_id,
                        "task depends on itself",
                        {"dependency": dependency},
                    )
                )
            elif dependency not in known_ids:
                issues.append(
                    PlanLinkIssue(
                        "TASK_DEPENDENCY_UNKNOWN",
                        task_id,
                        "task depends on an unknown task",
                        {"dependency": dependency},
                    )
                )
            else:
                edges.append((dependency, task_id))

    cycle = _cycle_nodes(tuple(known_ids), edges)
    if cycle:
        issues.append(
            PlanLinkIssue(
                "TASK_DEPENDENCY_CYCLE",
                "",
                "task dependency graph contains a cycle",
                {"cycle_nodes": cycle},
            )
        )

    production_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    raw_modules = handoff.get("production_modules")
    modules = raw_modules if isinstance(raw_modules, list) else []
    seen_module_ids: set[str] = set()
    for index, raw in enumerate(modules):
        if not isinstance(raw, Mapping):
            issues.append(
                PlanLinkIssue(
                    "PRODUCTION_BINDING_NOT_OBJECT",
                    "",
                    "production binding must be an object",
                    {"binding_index": index},
                )
            )
            continue
        task_ref = str(raw.get("task_ref") or "").strip()
        binding_id = str(raw.get("production_module_id") or "").strip()
        if not binding_id:
            issues.append(
                PlanLinkIssue(
                    "PRODUCTION_BINDING_ID_MISSING",
                    task_ref,
                    "production binding has no stable id",
                    {"binding_index": index},
                )
            )
        elif binding_id in seen_module_ids:
            issues.append(
                PlanLinkIssue(
                    "PRODUCTION_BINDING_ID_DUPLICATE",
                    task_ref,
                    "production binding id is duplicated",
                    {"production_module_id": binding_id},
                )
            )
        seen_module_ids.add(binding_id)
        if task_ref not in known_ids:
            issues.append(
                PlanLinkIssue(
                    "PRODUCTION_BINDING_TASK_UNKNOWN",
                    task_ref,
                    "production binding references an unknown task",
                    {"production_module_id": binding_id},
                )
            )
        else:
            production_by_task[task_ref].append(raw)

    assets_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    raw_assets = handoff.get("asset_requests")
    assets = raw_assets if isinstance(raw_assets, list) else []
    for index, raw in enumerate(assets):
        if not isinstance(raw, Mapping):
            issues.append(
                PlanLinkIssue(
                    "ASSET_BINDING_NOT_OBJECT",
                    "",
                    "asset binding must be an object",
                    {"binding_index": index},
                )
            )
            continue
        task_ref = str(raw.get("task_ref") or "").strip()
        if task_ref not in known_ids:
            issues.append(
                PlanLinkIssue(
                    "ASSET_BINDING_TASK_UNKNOWN",
                    task_ref,
                    "asset binding references an unknown task",
                    {"asset_request_id": str(raw.get("asset_request_id") or "")},
                )
            )
        else:
            assets_by_task[task_ref].append(raw)

    for task_id, task in task_items:
        anchors = _anchors(task)
        anchor_keys = {
            (str(anchor.get("kind") or ""), _locator(anchor))
            for anchor in anchors
            if _locator(anchor)
        }
        source_anchors = tuple(anchor for anchor in anchors if _is_source_symbol(anchor))
        test_anchors = tuple(anchor for anchor in anchors if _is_test_path(anchor))
        gates = frozenset(_strings(task.get("required_gates")))
        bindings = production_by_task.get(task_id, [])
        asset_bindings = assets_by_task.get(task_id, [])

        if not bindings and not asset_bindings:
            issues.append(
                PlanLinkIssue(
                    "TASK_EXECUTABLE_BINDING_MISSING",
                    task_id,
                    "executable task lowered to neither production nor asset binding",
                    {"owned_anchor_count": len(anchors)},
                )
            )

        if _claims_runtime(task) and test_anchors and not source_anchors and not asset_bindings:
            issues.append(
                PlanLinkIssue(
                    "TASK_RUNTIME_TEST_ONLY",
                    task_id,
                    "runtime capability is backed only by test/GameTest paths",
                    {"test_locators": [_locator(item) for item in test_anchors]},
                )
            )

        if gates & _COMPILE_GATES and not source_anchors:
            issues.append(
                PlanLinkIssue(
                    "TASK_SOURCE_BINDING_MISSING",
                    task_id,
                    "source/compile gate requires a concrete production Java symbol",
                    {"required_gates": sorted(gates)},
                )
            )

        for anchor in source_anchors:
            status = str(anchor.get("status") or "").casefold()
            if status != "host_reserved":
                issues.append(
                    PlanLinkIssue(
                        "TASK_SOURCE_NOT_RESERVED",
                        task_id,
                        "planned production source path is not host_reserved",
                        {"locator": _locator(anchor), "status": status},
                    )
                )

        for binding in bindings:
            bound_anchors = binding.get("owned_anchors")
            if not isinstance(bound_anchors, Sequence) or isinstance(
                bound_anchors, (str, bytes, bytearray)
            ):
                issues.append(
                    PlanLinkIssue(
                        "PRODUCTION_BINDING_ANCHORS_MISSING",
                        task_id,
                        "production binding has no concrete owned_anchors",
                        {"production_module_id": str(binding.get("production_module_id") or "")},
                    )
                )
                continue
            for raw_anchor in bound_anchors:
                if not isinstance(raw_anchor, Mapping):
                    issues.append(
                        PlanLinkIssue(
                            "PRODUCTION_BINDING_ANCHOR_INVALID",
                            task_id,
                            "production binding anchor is not an object",
                            {},
                        )
                    )
                    continue
                key = (str(raw_anchor.get("kind") or ""), _locator(raw_anchor))
                if key not in anchor_keys:
                    issues.append(
                        PlanLinkIssue(
                            "PRODUCTION_BINDING_NOT_OWNED",
                            task_id,
                            "production binding anchor is not present in task owned_anchors",
                            {"kind": key[0], "locator": key[1]},
                        )
                    )

        reuse_actions = {
            str(binding.get("reuse_action") or "").strip().casefold()
            for binding in [*bindings, *asset_bindings]
            if str(binding.get("reuse_action") or "").strip()
        }
        if len(reuse_actions) > 1:
            issues.append(
                PlanLinkIssue(
                    "TASK_REUSE_ACTION_CONFLICT",
                    task_id,
                    "production and asset bindings disagree on reuse action",
                    {"reuse_actions": sorted(reuse_actions)},
                )
            )
        reuse_refs = _strings(task.get("reuse_refs"))
        if reuse_actions == {"fresh"} and reuse_refs:
            issues.append(
                PlanLinkIssue(
                    "TASK_FRESH_HAS_REUSE_REFS",
                    task_id,
                    "fresh task carries donor/reuse refs",
                    {"reuse_refs": reuse_refs},
                )
            )
        if reuse_actions == {"adapt"} and not reuse_refs:
            issues.append(
                PlanLinkIssue(
                    "TASK_ADAPT_REUSE_REFS_MISSING",
                    task_id,
                    "adapt task has no verified donor/reuse refs",
                    {},
                )
            )

    return tuple(issues)


def validate_plan_collect_all(
    plan: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> None:
    issues = collect_plan_link_issues(plan, handoff)
    if not issues:
        emit_root_cause(
            "plan_collect_all_preflight",
            stage="planning",
            operation="validate_plan_collect_all",
            gate="production_linker",
            result="PASS",
            details={
                "task_count": len(plan.get("tasks") or ()),
                "issue_count": 0,
            },
        )
        return
    error = PlanCollectAllLinkError(issues)
    emit_root_cause(
        "plan_collect_all_preflight",
        stage="planning",
        operation="validate_plan_collect_all",
        gate="production_linker",
        result="FAIL",
        reason=str(error),
        details={
            "task_count": len(plan.get("tasks") or ()),
            "issue_count": len(issues),
            "issues": [item.to_dict() for item in issues],
        },
        exc=error,
    )
    raise error


__all__ = [
    "PlanCollectAllLinkError",
    "PlanLinkIssue",
    "collect_plan_link_issues",
    "validate_plan_collect_all",
]
