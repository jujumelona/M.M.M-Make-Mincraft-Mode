from __future__ import annotations

"""Inference-only procedural memory: verified trajectories -> ephemeral skill."""

import json
import os
import threading
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence

from .remote_trajectory_store import (
    flush_remote_outbox,
    hydrate_remote_cache,
    queue_remote_record,
    remote_configured,
)
from .trajectory_memory import (
    append_trajectory,
    build_work_trajectory,
    execution_context_from_messages,
    execution_context_from_values,
    memory_path,
    relevant_trajectories,
    remote_cache_path,
    synthesize_temporary_skill,
    task_class_for_stage,
)

_PREFIX = "MMM TEMPORARY VERIFIED SKILL:\n"
_CACHE_LOCK = threading.RLock()


def _query(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for message in reversed(messages):
        if str(message.get("role", "")).casefold() not in {"user", "system", "tool"}:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        if sum(len(item) for item in parts) >= 8000:
            break
    return "\n".join(reversed(parts))[-8000:]


def _inject(messages: Sequence[Mapping[str, Any]], skill: Mapping[str, Any]) -> list[dict[str, Any]]:
    rendered = _PREFIX + json.dumps(skill, ensure_ascii=False, sort_keys=True)
    result = [
        dict(message)
        for message in messages
        if not (
            str(message.get("role", "")) == "system"
            and isinstance(message.get("content"), str)
            and str(message.get("content", "")).startswith(_PREFIX)
        )
    ]
    insertion = 0
    while insertion < len(result) and str(result[insertion].get("role", "")) == "system":
        insertion += 1
    result.insert(insertion, {"role": "system", "content": rendered})
    return result


def _skill_cache_capacity() -> int:
    raw = os.environ.get("MMM_TEMPORARY_SKILL_CACHE_ENTRIES", "64").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 64
    return max(0, min(value, 512))


def _path_fingerprint(path: Path) -> tuple[Any, ...] | None:
    """Return a strict local-file identity, or None when safe reuse is impossible."""

    try:
        if path.is_symlink():
            return None
        stat = path.stat()
    except FileNotFoundError:
        return ("missing",)
    except OSError:
        return None
    if not path.is_file():
        return None
    return (
        "file",
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _trajectory_fingerprint(root: Path, task_class: str) -> tuple[Any, ...] | None:
    local = _path_fingerprint(memory_path(root))
    remote = _path_fingerprint(remote_cache_path(root, task_class))
    if local is None or remote is None:
        return None
    return (local, remote)


def _temporary_skill(
    router: Any,
    root: Path,
    query: str,
    *,
    task_class: str,
    limit: int = 6,
    current_context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Reuse only a skill from the unchanged corpus and compatible current state."""

    capacity = _skill_cache_capacity()
    before = _trajectory_fingerprint(root, task_class)
    normalized_context = execution_context_from_values(current_context or {})
    context_key = json.dumps(
        normalized_context,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    key = (str(root), task_class, query, int(limit), context_key, before)
    if capacity > 0 and before is not None:
        with _CACHE_LOCK:
            cache = getattr(router, "_mmm_temporary_skill_cache", None)
            if isinstance(cache, OrderedDict) and key in cache:
                skill = cache.pop(key)
                cache[key] = skill
                return skill

    records = relevant_trajectories(
        root,
        query,
        task_class=task_class,
        router=router,
        limit=limit,
        current_context=normalized_context,
    )
    skill = synthesize_temporary_skill(query, records, task_class=task_class)

    after = _trajectory_fingerprint(root, task_class)
    if capacity > 0 and before is not None and before == after:
        with _CACHE_LOCK:
            cache = getattr(router, "_mmm_temporary_skill_cache", None)
            if not isinstance(cache, OrderedDict):
                cache = OrderedDict()
                router._mmm_temporary_skill_cache = cache
            cache[key] = skill
            cache.move_to_end(key)
            while len(cache) > capacity:
                cache.popitem(last=False)
    return skill


def _canonical_read_key(call: Any) -> tuple[str, str] | None:
    try:
        name = str(call.name)
        arguments = json.dumps(
            dict(call.arguments),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return name, arguments


def _install_read_wave_dedup(model_router_module: Any) -> None:
    """Single-flight exact duplicate read tools inside one mutation-free wave."""

    current = model_router_module._execute_tool_waves
    if getattr(current, "_mmm_exact_read_wave_dedup", False):
        return

    parallel_reads = frozenset(model_router_module._PARALLEL_READ_TOOLS)

    @wraps(current)
    def execute_with_dedup(
        calls: Sequence[Any],
        execute: Any,
    ) -> tuple[tuple[Any, Mapping[str, Any]], ...]:
        completed: list[tuple[Any, Mapping[str, Any]]] = []
        pending_reads: list[Any] = []

        def flush_reads() -> None:
            if not pending_reads:
                return
            batch = tuple(pending_reads)
            pending_reads.clear()

            unique: list[Any] = []
            representative_index: list[int] = []
            exact: dict[tuple[str, str], int] = {}
            for call in batch:
                key = _canonical_read_key(call)
                if key is None:
                    representative_index.append(len(unique))
                    unique.append(call)
                    continue
                index = exact.get(key)
                if index is None:
                    index = len(unique)
                    exact[key] = index
                    unique.append(call)
                representative_index.append(index)

            executed_unique = tuple(current(tuple(unique), execute))
            if len(executed_unique) != len(unique):
                raise RuntimeError(
                    "Tool-wave executor returned a result count that does not match "
                    "the deduplicated read wave."
                )
            for call, index in zip(batch, representative_index, strict=True):
                _representative, payload = executed_unique[index]
                completed.append((call, payload))

        for call in calls:
            if call.name in parallel_reads:
                pending_reads.append(call)
                continue
            flush_reads()
            completed.append(execute(call))
        flush_reads()
        return tuple(completed)

    execute_with_dedup._mmm_exact_read_wave_dedup = True  # type: ignore[attr-defined]
    execute_with_dedup.__wrapped__ = current  # type: ignore[attr-defined]
    model_router_module._execute_tool_waves = execute_with_dedup


def _install_model_skill(model_router_module: Any) -> None:
    cls = model_router_module.ModelRouter
    current = cls._prepare_generation_request
    if getattr(current, "_mmm_temporary_verified_skill", False):
        return

    @wraps(current)
    def prepare_with_skill(self: Any, role: str, messages: Sequence[Mapping[str, Any]], **kwargs: Any):
        stage, runtime, tools, request = current(self, role, messages, **kwargs)
        root_value = getattr(self, "_agent_workspace_root", None)
        if root_value is None:
            return stage, runtime, tools, request
        root = Path(root_value).expanduser().resolve()
        task_class = task_class_for_stage(stage or role)
        hydrated = getattr(self, "_mmm_remote_skill_hydrated", None)
        if not isinstance(hydrated, set):
            hydrated = set()
            self._mmm_remote_skill_hydrated = hydrated
        if remote_configured() and task_class not in hydrated:
            hydrate_remote_cache(root, task_class)
            hydrated.add(task_class)
        query = _query(messages)
        current_context = execution_context_from_messages(messages)
        skill = _temporary_skill(
            self,
            root,
            query,
            task_class=task_class,
            limit=6,
            current_context=current_context,
        )
        qualified_count = (
            len(skill.get("source_trajectory_ids", ()))
            if isinstance(skill, Mapping)
            else 0
        )
        if skill is None or qualified_count < 2:
            return stage, runtime, tools, request

        from .model_adapters import GenerationRequest

        rebuilt = GenerationRequest(
            messages=_inject(request.messages, skill),
            media_paths=request.media_paths,
            response_format=request.response_format,
            response_schema=request.response_schema,
            tools=request.tools,
            tool_choice=request.tool_choice,
            parallel_tool_calls=request.parallel_tool_calls,
        )
        print(
            "temporary skill:",
            f"class={task_class}",
            f"qualified_trajectories={qualified_count}",
            f"patterns={len(skill.get('proven_patterns', ()))}/{len(skill.get('avoid_patterns', ())) }",
            f"context_keys={len(current_context)}",
            flush=True,
        )
        return stage, runtime, tools, rebuilt

    prepare_with_skill._mmm_temporary_verified_skill = True  # type: ignore[attr-defined]
    prepare_with_skill.__wrapped__ = current  # type: ignore[attr-defined]
    cls._prepare_generation_request = prepare_with_skill


def _record(base: Path, row: Mapping[str, Any]) -> None:
    append_trajectory(base, row)
    queue_remote_record(base, row)


def _install_work_trajectory(work_graph_module: Any) -> None:
    cls = work_graph_module.DurableWorkLedger
    current_succeed = cls.succeed
    if not getattr(current_succeed, "_mmm_verified_work_trajectory", False):
        @wraps(current_succeed)
        def succeed_with_trajectory(self: Any, node_id: str, receipt: dict[str, Any], *, output_hash: str = ""):
            result = current_succeed(self, node_id, receipt, output_hash=output_hash)
            task = result if isinstance(result, Mapping) else self.task(node_id)
            row = build_work_trajectory(task, outcome="SUCCESS", receipt=receipt)
            base = Path(self.path).resolve().parent
            _record(base, row)
            if task_class_for_stage(str(task.get("stage", ""))) == "release":
                try:
                    sync = flush_remote_outbox(base) if remote_configured() else None
                    if sync:
                        print("trajectory remote sync:", sync.get("status"), sync.get("flushed", 0), flush=True)
                except Exception as exc:
                    print("trajectory remote sync deferred:", f"{type(exc).__name__}: {str(exc)[:240]}", flush=True)
            return result

        succeed_with_trajectory._mmm_verified_work_trajectory = True  # type: ignore[attr-defined]
        succeed_with_trajectory.__wrapped__ = current_succeed  # type: ignore[attr-defined]
        cls.succeed = succeed_with_trajectory

    current_fail = cls.fail
    if not getattr(current_fail, "_mmm_failed_work_trajectory", False):
        @wraps(current_fail)
        def fail_with_trajectory(self: Any, node_id: str, error: str, *, input_required: bool = False):
            result = current_fail(self, node_id, error, input_required=input_required)
            task = result if isinstance(result, Mapping) else self.task(node_id)
            row = build_work_trajectory(task, outcome="FAIL", error=error)
            _record(Path(self.path).resolve().parent, row)
            return result

        fail_with_trajectory._mmm_failed_work_trajectory = True  # type: ignore[attr-defined]
        fail_with_trajectory.__wrapped__ = current_fail  # type: ignore[attr-defined]
        cls.fail = fail_with_trajectory


def _install_repair_trajectory(repair_module: Any) -> None:
    cls = repair_module.RepairEngine
    current = cls.repair
    if getattr(current, "_mmm_verified_repair_trajectory", False):
        return

    @wraps(current)
    def repair_with_trajectory(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        result = current(self, project_root, *args, **kwargs)
        if not isinstance(result, Mapping):
            return result
        status = str(result.get("status", ""))
        task = {
            "node_id": "repair",
            "stage": "repair",
            "payload": {"kind": "repair"},
        }
        row = build_work_trajectory(
            task,
            outcome="SUCCESS" if status == "PASS" else "FAIL",
            receipt=result,
            error=str(result.get("error", "")),
        )
        _record(Path(project_root).expanduser().resolve(), row)
        return result

    repair_with_trajectory._mmm_verified_repair_trajectory = True  # type: ignore[attr-defined]
    repair_with_trajectory.__wrapped__ = current  # type: ignore[attr-defined]
    cls.repair = repair_with_trajectory


def install(*, model_router_module: Any, work_graph_module: Any, repair_module: Any) -> None:
    _install_read_wave_dedup(model_router_module)
    _install_model_skill(model_router_module)
    _install_work_trajectory(work_graph_module)
    _install_repair_trajectory(repair_module)


__all__ = ["install"]
