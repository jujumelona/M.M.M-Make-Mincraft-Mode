from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


EXECUTION_SCHEMA = "mmm/evidence-first-execution-v1"
ERROR_OBSERVATION_SCHEMA = "mmm/action-repair-observation-v1"
MAX_LOCAL_CONTEXT_BYTES = 8192
_FORBIDDEN_CONTEXT_KEYS = {
    "conversation",
    "full_prompt",
    "messages",
    "prompt",
    "request",
    "request_catalog",
    "system_prompt",
}


class EvidenceExecutionError(ValueError):
    """Raised when evidence-first execution would violate a bounded contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_text(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise EvidenceExecutionError(f"{name} must be non-empty")
    return text


def _normalize_path(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.strip("/")


def _validate_local_context(value: Any) -> Any:
    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key).strip().casefold()
                if key in _FORBIDDEN_CONTEXT_KEYS:
                    raise EvidenceExecutionError(
                        f"local repair context must not include global field {raw_key!r}"
                    )
                walk(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(value)
    frozen = deepcopy(value)
    size = len(_canonical_json(frozen).encode("utf-8"))
    if size > MAX_LOCAL_CONTEXT_BYTES:
        raise EvidenceExecutionError(
            f"local repair context exceeds {MAX_LOCAL_CONTEXT_BYTES} bytes"
        )
    return frozen


def build_local_error_observation(
    *,
    task_id: str,
    action_id: str,
    failed_action: Mapping[str, Any],
    json_pointer: str,
    code: str,
    message: str,
    allowed_keys: Iterable[str] = (),
    local_context: Any = None,
) -> dict[str, Any]:
    """Create the only payload a local action-repair callback should receive.

    The observation intentionally excludes the original request, prompt and catalog.
    Callers may include a small malformed argument fragment in ``local_context``.
    """

    context = _validate_local_context({} if local_context is None else local_context)
    payload: dict[str, Any] = {
        "schema": ERROR_OBSERVATION_SCHEMA,
        "task_id": _require_text(task_id, "task_id"),
        "action_id": _require_text(action_id, "action_id"),
        "error": {
            "json_pointer": _require_text(json_pointer, "json_pointer"),
            "code": _require_text(code, "code"),
            "message": _require_text(message, "message"),
            "allowed_keys": sorted(
                {str(item).strip() for item in allowed_keys if str(item).strip()}
            ),
        },
        "local_context": context,
        "failed_action_digest": _sha256(dict(failed_action)),
    }
    payload["observation_sha256"] = _sha256(payload)
    return payload


def _assert_action_identity(action: Mapping[str, Any], *, task_id: str, action_id: str) -> None:
    if "task_id" in action and str(action["task_id"]) != task_id:
        raise EvidenceExecutionError("local repair changed task_id")
    if "action_id" in action and str(action["action_id"]) != action_id:
        raise EvidenceExecutionError("local repair changed action_id")


def run_local_action_repair_loop(
    *,
    task_id: str,
    action_id: str,
    initial_action: Mapping[str, Any],
    validate_action: Callable[[Mapping[str, Any]], None],
    execute_action: Callable[[Mapping[str, Any]], Any],
    describe_error: Callable[[Exception, Mapping[str, Any]], Mapping[str, Any]],
    repair_action: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    max_attempts: int = 2,
) -> dict[str, Any]:
    """Validate/execute one action and repair it locally with bounded observations only."""

    task_ref = _require_text(task_id, "task_id")
    action_ref = _require_text(action_id, "action_id")
    if max_attempts < 1:
        raise EvidenceExecutionError("max_attempts must be >= 1")

    action: Mapping[str, Any] = deepcopy(dict(initial_action))
    _assert_action_identity(action, task_id=task_ref, action_id=action_ref)
    observations: list[dict[str, Any]] = []

    for attempt in range(1, max_attempts + 1):
        try:
            validate_action(action)
            result = execute_action(action)
            return {
                "schema": EXECUTION_SCHEMA,
                "task_id": task_ref,
                "action_id": action_ref,
                "attempts": attempt,
                "observations": observations,
                "action": deepcopy(dict(action)),
                "result": result,
            }
        except Exception as exc:  # repair policy is supplied by the caller
            if attempt >= max_attempts:
                raise EvidenceExecutionError(
                    f"action {action_ref} failed after {max_attempts} bounded attempt(s): {exc}"
                ) from exc
            descriptor = dict(describe_error(exc, action))
            observation = build_local_error_observation(
                task_id=task_ref,
                action_id=action_ref,
                failed_action=action,
                json_pointer=str(descriptor.get("json_pointer") or "/"),
                code=str(descriptor.get("code") or type(exc).__name__),
                message=str(descriptor.get("message") or exc),
                allowed_keys=descriptor.get("allowed_keys") or (),
                local_context=descriptor.get("local_context") or {},
            )
            observations.append(observation)
            repaired = repair_action(deepcopy(observation))
            if not isinstance(repaired, Mapping):
                raise EvidenceExecutionError("local repair must return an action mapping")
            _assert_action_identity(repaired, task_id=task_ref, action_id=action_ref)
            action = deepcopy(dict(repaired))

    raise AssertionError("unreachable")


@dataclass
class EvidenceExecutionState:
    path: Path
    plan_sha256: str
    completed_task_ids: list[str] = field(default_factory=list)
    task_receipts: dict[str, Any] = field(default_factory=dict)
    impact_events: list[dict[str, Any]] = field(default_factory=list)
    last_index_sha256: str = ""

    @classmethod
    def load_or_create(cls, path: str | Path, *, plan_sha256: str) -> "EvidenceExecutionState":
        state_path = Path(path)
        plan_ref = _require_text(plan_sha256, "plan_sha256")
        if not state_path.exists():
            return cls(path=state_path, plan_sha256=plan_ref)
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceExecutionError(f"invalid execution checkpoint: {exc}") from exc
        if payload.get("schema") != EXECUTION_SCHEMA:
            raise EvidenceExecutionError("execution checkpoint schema mismatch")
        if str(payload.get("plan_sha256") or "") != plan_ref:
            raise EvidenceExecutionError("execution checkpoint belongs to a different plan")
        return cls(
            path=state_path,
            plan_sha256=plan_ref,
            completed_task_ids=[str(item) for item in payload.get("completed_task_ids") or []],
            task_receipts=dict(payload.get("task_receipts") or {}),
            impact_events=[dict(item) for item in payload.get("impact_events") or []],
            last_index_sha256=str(payload.get("last_index_sha256") or ""),
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": EXECUTION_SCHEMA,
            "plan_sha256": self.plan_sha256,
            "completed_task_ids": list(dict.fromkeys(self.completed_task_ids)),
            "task_receipts": deepcopy(self.task_receipts),
            "impact_events": deepcopy(self.impact_events),
            "last_index_sha256": self.last_index_sha256,
        }
        payload["state_sha256"] = _sha256(payload)
        return payload

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(_canonical_json(self.as_dict()) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def checkpoint_success(self, task_id: str, receipt: Any) -> None:
        task_ref = _require_text(task_id, "task_id")
        if task_ref not in self.completed_task_ids:
            self.completed_task_ids.append(task_ref)
        self.task_receipts[task_ref] = deepcopy(receipt)
        self.flush()

    def record_impact_event(self, event: Mapping[str, Any], *, index_sha256: str) -> None:
        frozen = deepcopy(dict(event))
        frozen["event_sha256"] = _sha256(frozen)
        self.impact_events.append(frozen)
        self.last_index_sha256 = str(index_sha256 or "")
        self.flush()


@dataclass(frozen=True)
class ProjectIndexRefresh:
    previous_sha256: str
    current_sha256: str
    changed_paths: tuple[str, ...]
    index: Mapping[str, Any]


def _index_file_map(index: Mapping[str, Any] | None) -> dict[str, str]:
    if not index:
        return {}
    raw_files = index.get("files") or {}
    result: dict[str, str] = {}
    if isinstance(raw_files, Mapping):
        for raw_path, value in raw_files.items():
            path = _normalize_path(raw_path)
            if not path:
                continue
            if isinstance(value, Mapping):
                digest = value.get("sha256") or value.get("digest") or value.get("hash") or value
            else:
                digest = value
            result[path] = _sha256(digest) if not isinstance(digest, str) else digest
    elif isinstance(raw_files, Sequence) and not isinstance(raw_files, (str, bytes)):
        for item in raw_files:
            if not isinstance(item, Mapping):
                continue
            path = _normalize_path(item.get("path") or item.get("locator"))
            if not path:
                continue
            digest = item.get("sha256") or item.get("digest") or item.get("hash") or item
            result[path] = _sha256(digest) if not isinstance(digest, str) else digest
    return result


def refresh_project_index(
    previous_index: Mapping[str, Any] | None,
    *,
    index_builder: Callable[[], Mapping[str, Any]],
) -> ProjectIndexRefresh:
    current = deepcopy(dict(index_builder()))
    previous = deepcopy(dict(previous_index or {}))
    before = _index_file_map(previous)
    after = _index_file_map(current)
    changed = sorted(
        path
        for path in set(before) | set(after)
        if before.get(path) != after.get(path)
    )
    return ProjectIndexRefresh(
        previous_sha256=_sha256(previous),
        current_sha256=_sha256(current),
        changed_paths=tuple(changed),
        index=current,
    )


def _ownership_paths(task: Mapping[str, Any]) -> set[str]:
    paths = {
        _normalize_path(item)
        for item in task.get("owned_paths") or []
        if _normalize_path(item)
    }
    for anchor in task.get("ownership") or task.get("owned_anchors") or []:
        if not isinstance(anchor, Mapping):
            continue
        kind = str(anchor.get("kind") or "").casefold()
        locator = _normalize_path(anchor.get("locator"))
        if locator and kind in {"file", "path", "resource", "source", "source_file"}:
            paths.add(locator)
    return paths


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(right + "/") or right.startswith(left + "/")


def downstream_task_ids(
    tasks: Sequence[Mapping[str, Any]],
    seed_task_ids: Iterable[str],
) -> list[str]:
    ordered_ids = [_require_text(task.get("task_id"), "task_id") for task in tasks]
    known = set(ordered_ids)
    seeds = {str(item) for item in seed_task_ids}
    unknown = seeds - known
    if unknown:
        raise EvidenceExecutionError(f"unknown impact seed task(s): {sorted(unknown)}")
    impacted = set(seeds)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            task_id = str(task.get("task_id") or "")
            dependencies = {str(item) for item in task.get("depends_on") or []}
            if task_id not in impacted and dependencies & impacted:
                impacted.add(task_id)
                changed = True
    return [task_id for task_id in ordered_ids if task_id in impacted]


def impacted_task_ids_for_paths(
    tasks: Sequence[Mapping[str, Any]],
    changed_paths: Iterable[str],
    *,
    completed_task_ids: Iterable[str] = (),
) -> list[str]:
    paths = {_normalize_path(path) for path in changed_paths if _normalize_path(path)}
    seeds: list[str] = []
    for task in tasks:
        ownership = _ownership_paths(task)
        if any(_paths_overlap(path, owned) for path in paths for owned in ownership):
            seeds.append(str(task.get("task_id") or ""))
    if not seeds:
        return []
    completed = {str(item) for item in completed_task_ids}
    return [
        task_id
        for task_id in downstream_task_ids(tasks, seeds)
        if task_id not in completed
    ]


def checkpoint_refresh_and_replan(
    *,
    state: EvidenceExecutionState,
    completed_task_id: str,
    success_receipt: Any,
    tasks: Sequence[Mapping[str, Any]],
    previous_index: Mapping[str, Any] | None,
    index_builder: Callable[[], Mapping[str, Any]],
    replanner: Callable[[Sequence[Mapping[str, Any]]], Any],
) -> dict[str, Any]:
    """Durably checkpoint success, refresh project evidence, then replan only impact."""

    state.checkpoint_success(completed_task_id, success_receipt)
    refresh = refresh_project_index(previous_index, index_builder=index_builder)
    impacted_ids = impacted_task_ids_for_paths(
        tasks,
        refresh.changed_paths,
        completed_task_ids=state.completed_task_ids,
    )
    impacted_set = set(impacted_ids)
    impacted_tasks = [deepcopy(dict(task)) for task in tasks if str(task.get("task_id")) in impacted_set]
    replan_result = replanner(impacted_tasks) if impacted_tasks else None
    event = {
        "completed_task_id": _require_text(completed_task_id, "completed_task_id"),
        "changed_paths": list(refresh.changed_paths),
        "impacted_task_ids": impacted_ids,
        "previous_index_sha256": refresh.previous_sha256,
        "current_index_sha256": refresh.current_sha256,
        "replanned": bool(impacted_tasks),
    }
    state.record_impact_event(event, index_sha256=refresh.current_sha256)
    return {
        "refresh": refresh,
        "impacted_task_ids": impacted_ids,
        "replan_result": replan_result,
    }


__all__ = [
    "ERROR_OBSERVATION_SCHEMA",
    "EXECUTION_SCHEMA",
    "EvidenceExecutionError",
    "EvidenceExecutionState",
    "ProjectIndexRefresh",
    "build_local_error_observation",
    "checkpoint_refresh_and_replan",
    "downstream_task_ids",
    "impacted_task_ids_for_paths",
    "refresh_project_index",
    "run_local_action_repair_loop",
]
