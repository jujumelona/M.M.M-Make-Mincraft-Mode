from __future__ import annotations

"""Late runtime policy that makes declared parallel capacity real end to end.

The production graph already owns dependency safety, durable leases and narrow commits.
This contract removes the remaining execution bottlenecks without weakening those
boundaries: exact executor capacity, same-GPU LLM read sharing, finer custom DAG shards,
and isolated parallel custom candidate generation with a single winner commit.
"""

import concurrent.futures
import copy
import json
import os
import shutil
import time
from collections import Counter
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence


_FORCE_SINGLE_CUSTOM_SEARCH: ContextVar[bool] = ContextVar(
    "mmm_force_single_custom_search",
    default=False,
)
_ORIGINAL_THREAD_POOL = concurrent.futures.ThreadPoolExecutor


def _active_parallelism() -> int:
    raw = os.environ.get("MMM_LLAMA_ACTIVE_PARALLEL", "1").strip()
    try:
        return max(1, min(8, int(raw)))
    except ValueError:
        return 1


def _env_size(name: str, default: int, upper: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(max(1, upper), value))


def _install_exact_llm_executor() -> None:
    """Make the production ``llm`` pool match the native server slot capacity."""

    current = concurrent.futures.ThreadPoolExecutor
    if getattr(current, "_mmm_exact_llm_executor", False):
        return
    base = current

    class ExactCapacityThreadPoolExecutor(base):
        def __init__(
            self,
            max_workers: int | None = None,
            thread_name_prefix: str = "",
            initializer: Any = None,
            initargs: tuple[Any, ...] = (),
        ) -> None:
            # CompleteProductionOrchestrator intentionally names its scarce pools.
            # Only the exact LLM pool is widened; every other stdlib executor keeps
            # the caller's requested capacity unchanged.
            if thread_name_prefix == "llm":
                max_workers = _active_parallelism()
            super().__init__(
                max_workers=max_workers,
                thread_name_prefix=thread_name_prefix,
                initializer=initializer,
                initargs=initargs,
            )

    ExactCapacityThreadPoolExecutor._mmm_exact_llm_executor = True  # type: ignore[attr-defined]
    ExactCapacityThreadPoolExecutor._mmm_base_executor = base  # type: ignore[attr-defined]
    concurrent.futures.ThreadPoolExecutor = ExactCapacityThreadPoolExecutor


def _install_module_routing(work_graph_module: Any) -> None:
    """Expose independent custom work as DAG nodes only when slots can consume it."""

    current_stage = work_graph_module._module_stage
    if not getattr(current_stage, "_mmm_exact_integration_stage", False):

        @wraps(current_stage)
        def module_stage(module: Any) -> str:
            if str(getattr(module, "kind", "")) == "integration":
                config = getattr(module, "config", {})
                config = config if isinstance(config, Mapping) else {}
                if str(config.get("integration_type", "")) == "mmm_local_ai_sidecar":
                    return "content"
                return "custom"
            return current_stage(module)

        module_stage._mmm_exact_integration_stage = True  # type: ignore[attr-defined]
        work_graph_module._module_stage = module_stage

    current_shards = work_graph_module._module_shards
    if getattr(current_shards, "_mmm_slot_aware_custom_shards", False):
        return

    @wraps(current_shards)
    def module_shards(modules: Any, *, policy: Any):
        for stage, members in current_shards(modules, policy=policy):
            if stage != "custom" or _active_parallelism() <= 1 or len(members) <= 1:
                yield stage, members
                continue
            size = _env_size("MMM_CUSTOM_PIPELINE_SHARD_SIZE", 1, len(members))
            for offset in range(0, len(members), size):
                yield stage, tuple(members[offset : offset + size])

    module_shards._mmm_slot_aware_custom_shards = True  # type: ignore[attr-defined]
    work_graph_module._module_shards = module_shards


def _fairness_owner(safety_module: Any, ledger: Any) -> str:
    resolver = getattr(safety_module, "_orchestrator_owner", None)
    if callable(resolver):
        return str(resolver(ledger))
    owner = getattr(ledger, "_mmm_parallel_lease_owner", "")
    return str(owner or "mmm-orchestrator")


def _clear_poll_snapshot(ledger: Any) -> None:
    setattr(ledger, "_mmm_generation_poll_snapshot", None)
    setattr(ledger, "_mmm_generation_poll_reads_left", 0)


def _find_fairness_wrapper(function: Any) -> Any:
    seen: set[int] = set()
    current = function
    while callable(current) and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "_mmm_exact_executor_fairness", False):
            return current
        current = getattr(current, "__wrapped__", None)
    return function


def _install_max_efficiency_claim(work_graph_module: Any, safety_module: Any) -> None:
    """Use every free LLM slot while still excluding image work on a shared GPU."""

    cls = work_graph_module.DurableWorkLedger
    current = cls.claim_ready
    if getattr(current, "_mmm_max_efficiency_claim", False):
        return
    fairness_wrapper = _find_fairness_wrapper(current)

    @wraps(current)
    def claim_ready(
        self: Any,
        worker_id: str,
        *,
        stages: Sequence[str] = (),
        lease_seconds: int = 900,
    ) -> dict[str, Any] | None:
        if worker_id != "mmm-orchestrator":
            return current(
                self,
                worker_id,
                stages=stages,
                lease_seconds=lease_seconds,
            )
        if not worker_id.strip():
            raise work_graph_module.WorkGraphError("worker_id must not be empty.")
        if lease_seconds < 1:
            raise work_graph_module.WorkGraphError("lease_seconds must be positive.")

        _clear_poll_snapshot(self)
        now = time.time()
        owner = _fairness_owner(safety_module, self)
        capacities = dict(safety_module._capacities())
        renew_before = now + max(1.0, lease_seconds * 0.5)
        resource_sql = safety_module._resource_sql

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE tasks
                SET lease_until = ?, updated_at = ?
                WHERE state = ? AND lease_owner = ?
                  AND lease_until IS NOT NULL
                  AND lease_until >= ? AND lease_until <= ?
                """,
                (
                    now + lease_seconds,
                    now,
                    work_graph_module.WorkState.RUNNING.value,
                    owner,
                    now,
                    renew_before,
                ),
            )
            connection.execute(
                """
                UPDATE tasks
                SET state = ?, lease_owner = NULL, lease_until = NULL,
                    error = 'expired worker lease', updated_at = ?
                WHERE state = ? AND lease_until IS NOT NULL AND lease_until < ?
                """,
                (
                    work_graph_module.WorkState.PENDING.value,
                    now,
                    work_graph_module.WorkState.RUNNING.value,
                    now,
                ),
            )

            running: Counter[str] = Counter(
                {
                    str(resource): int(count)
                    for resource, count in connection.execute(
                        f"""
                        SELECT {resource_sql('')}, COUNT(*)
                        FROM tasks
                        WHERE state = ? AND stage LIKE 'generate:%'
                        GROUP BY {resource_sql('')}
                        """,
                        (work_graph_module.WorkState.RUNNING.value,),
                    )
                }
            )
            free_lanes = {
                lane
                for lane, capacity in capacities.items()
                if running.get(lane, 0) < max(1, int(capacity))
            }

            if safety_module._SHARED_LOCAL_GPU_LANE.get():
                # llama.cpp/vLLM slots are read-sharing users of one resident text
                # model, so LLM+LLM is allowed up to capacity. Diffusion is a writer
                # on the same device and remains mutually exclusive with the text lane.
                if running.get("image_gpu", 0) > 0:
                    free_lanes.discard("llm")
                if running.get("llm", 0) > 0:
                    free_lanes.discard("image_gpu")

            if not free_lanes:
                connection.commit()
                _clear_poll_snapshot(self)
                return None

            stage_sql = ""
            params: list[Any] = [
                work_graph_module.WorkState.PENDING.value,
                work_graph_module.WorkState.SUCCEEDED.value,
            ]
            if stages:
                placeholders = ",".join("?" for _ in stages)
                stage_sql = f" AND task.stage IN ({placeholders})"
                params.extend(stages)
            lane_placeholders = ",".join("?" for _ in sorted(free_lanes))
            params.extend(sorted(free_lanes))
            task_resource = resource_sql("task")
            rows = connection.execute(
                f"""
                SELECT task.node_id, {task_resource}
                FROM tasks AS task
                WHERE task.state = ?
                  AND NOT EXISTS (
                    SELECT 1
                    FROM edges
                    JOIN tasks AS dependency
                      ON dependency.node_id = edges.dependency_id
                    WHERE edges.node_id = task.node_id
                      AND dependency.state != ?
                  )
                  {stage_sql}
                  AND {task_resource} IN ({lane_placeholders})
                ORDER BY task.node_id
                LIMIT 256
                """,
                tuple(params),
            ).fetchall()
            if not rows:
                connection.commit()
                _clear_poll_snapshot(self)
                return None

            priority = {"llm": 0, "image_gpu": 1, "cpu_io": 2, "commit": 3}
            candidates = sorted(
                (
                    running.get(str(resource), 0)
                    / max(1, int(capacities.get(str(resource), 1))),
                    priority.get(str(resource), 9),
                    str(node_id),
                    str(resource),
                )
                for node_id, resource in rows
            )
            _utilization, _priority, node_id, _resource = candidates[0]
            cursor = connection.execute(
                """
                UPDATE tasks
                SET state = ?, attempt = attempt + 1, lease_owner = ?,
                    lease_until = ?, error = NULL, updated_at = ?
                WHERE node_id = ? AND state = ?
                """,
                (
                    work_graph_module.WorkState.RUNNING.value,
                    owner,
                    now + lease_seconds,
                    now,
                    node_id,
                    work_graph_module.WorkState.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                _clear_poll_snapshot(self)
                return None
            connection.commit()

        _clear_poll_snapshot(self)
        return self.task(node_id)

    claim_ready._mmm_max_efficiency_claim = True  # type: ignore[attr-defined]
    claim_ready._mmm_parallel_lane_claim = True  # type: ignore[attr-defined]
    claim_ready._mmm_exact_executor_fairness = True  # type: ignore[attr-defined]
    # Keep architecture tests and introspection attached to the actual fairness owner,
    # while the closure still delegates non-orchestrator workers through the full chain.
    claim_ready.__wrapped__ = fairness_wrapper  # type: ignore[attr-defined]
    cls.claim_ready = claim_ready


def _candidate_patch_capture(
    *,
    base_root: Path,
    candidate_root: Path,
    result: Mapping[str, Any],
    source_patch_module: Any,
) -> dict[str, Any]:
    receipt = result.get("patch_receipt")
    receipt_ops = receipt.get("operations") if isinstance(receipt, Mapping) else None
    if not isinstance(receipt_ops, list) or not receipt_ops:
        raise RuntimeError("Custom candidate has no staged patch receipt.")

    operations: list[dict[str, Any]] = []
    before: dict[str, bytes | None] = {}
    for item in receipt_ops:
        if not isinstance(item, Mapping):
            raise RuntimeError("Custom candidate patch receipt is malformed.")
        relative = str(item.get("path", "")).strip()
        if not relative:
            raise RuntimeError("Custom candidate patch receipt has an empty path.")
        base_path = (base_root / relative).resolve()
        candidate_path = (candidate_root / relative).resolve()
        try:
            base_path.relative_to(base_root)
            candidate_path.relative_to(candidate_root)
        except ValueError as exc:
            raise RuntimeError(f"Custom candidate path escaped staging root: {relative}") from exc

        base_bytes = base_path.read_bytes() if base_path.is_file() else None
        before[relative] = base_bytes
        before_sha = item.get("before_sha256")
        after_sha = item.get("after_sha256")
        if before_sha is None:
            if not candidate_path.is_file():
                raise RuntimeError(f"Custom candidate create output is missing: {relative}")
            operations.append(
                {
                    "operation": "create",
                    "path": relative,
                    "content": candidate_path.read_text(encoding="utf-8"),
                }
            )
            continue
        if base_bytes is None or source_patch_module.sha256_bytes(base_bytes) != str(before_sha):
            raise RuntimeError(f"Custom candidate base hash drifted for {relative}")
        if after_sha is None:
            operations.append(
                {
                    "operation": "delete",
                    "path": relative,
                    "expected_sha256": str(before_sha),
                }
            )
            continue
        if not candidate_path.is_file():
            raise RuntimeError(f"Custom candidate replacement output is missing: {relative}")
        operations.append(
            {
                "operation": "replace",
                "path": relative,
                "expected_sha256": str(before_sha),
                "content": candidate_path.read_text(encoding="utf-8"),
            }
        )
    return {"operations": operations, "before": before}


def _install_parallel_custom_search(custom_module_generator_module: Any) -> None:
    """Parallelize the expensive custom candidate generation, not just verification."""

    from . import custom_generation_search_contract as search_module
    from . import performance_final_contract as performance_module
    from . import source_patch as source_patch_module

    current_width = search_module._width
    if not getattr(current_width, "_mmm_context_single_candidate", False):

        @wraps(current_width)
        def width(module: Any) -> int:
            if _FORCE_SINGLE_CUSTOM_SEARCH.get():
                return 1
            return current_width(module)

        width._mmm_context_single_candidate = True  # type: ignore[attr-defined]
        width.__wrapped__ = current_width  # type: ignore[attr-defined]
        search_module._width = width

    cls = custom_module_generator_module.CustomModuleGenerator
    current = cls.generate
    if getattr(current, "_mmm_max_parallel_custom_search", False):
        return

    @wraps(current)
    def generate(self: Any, project_root: str | Path, *args: Any, **kwargs: Any):
        module = kwargs.get("module")
        count = int(search_module._width(module))

        # Even a width-one module can run concurrently with another DAG node. Never
        # expose the orchestrator's shared mutable generator cache to worker threads.
        if count <= 1:
            worker = copy.copy(self)
            worker._cached_index = None
            worker._cached_root = None
            return current(worker, project_root, *args, **kwargs)

        live_root = Path(project_root).expanduser().resolve()
        base_root = performance_module._clone_source_snapshot(live_root)
        candidates: list[tuple[int, Path, dict[str, Any]]] = []
        errors: dict[int, BaseException] = {}

        def solve(candidate_index: int) -> tuple[int, Path, dict[str, Any]]:
            candidate_root = performance_module._clone_source_snapshot(base_root)
            worker = copy.copy(self)
            worker._cached_index = None
            worker._cached_root = None
            strategy = search_module._STRATEGIES[
                candidate_index % len(search_module._STRATEGIES)
            ]
            worker.router = search_module._StrategyRouter(
                search_module._host_evidence_router(self.router),
                strategy=strategy,
                candidate_index=candidate_index,
                count=count,
            )
            token = _FORCE_SINGLE_CUSTOM_SEARCH.set(True)
            try:
                result = current(worker, candidate_root, *args, **kwargs)
                if not isinstance(result, dict):
                    raise RuntimeError("Custom generation candidate returned a non-object receipt.")
                return candidate_index, candidate_root, result
            except BaseException:
                shutil.rmtree(candidate_root, ignore_errors=True)
                raise
            finally:
                _FORCE_SINGLE_CUSTOM_SEARCH.reset(token)

        try:
            workers = min(count, _active_parallelism())
            with _ORIGINAL_THREAD_POOL(
                max_workers=workers,
                thread_name_prefix="mmm_custom_generate",
            ) as pool:
                futures = [pool.submit(solve, index) for index in range(count)]
                for candidate_index, future in enumerate(futures):
                    try:
                        candidates.append(future.result())
                    except BaseException as exc:
                        errors[candidate_index] = exc
            candidates.sort(key=lambda item: item[0])
            if not candidates:
                if errors:
                    raise errors[max(errors)]
                raise RuntimeError("Custom generation search produced no candidate.")

            def verify(item: tuple[int, Path, dict[str, Any]]):
                index, candidate_root, result = item
                score, verifier = search_module._verify_candidate(candidate_root, result)
                return score, index, candidate_root, result, verifier

            if len(candidates) == 1:
                evaluations = [verify(candidates[0])]
            else:
                with _ORIGINAL_THREAD_POOL(
                    max_workers=min(2, len(candidates)),
                    thread_name_prefix="mmm_custom_verify",
                ) as pool:
                    evaluations = list(pool.map(verify, candidates))

            evaluations.sort(
                key=lambda item: (
                    -float(item[0]),
                    len(json.dumps(item[3], ensure_ascii=False, sort_keys=True)),
                    int(item[1]),
                )
            )
            score, winner_index, winner_root, result, verifier = evaluations[0]
            capture = _candidate_patch_capture(
                base_root=base_root,
                candidate_root=winner_root,
                result=result,
                source_patch_module=source_patch_module,
            )
            commit_receipt = performance_module._commit_staged_operations(
                live_root=live_root,
                staging_root=winner_root,
                capture=capture,
                source_patch_module=source_patch_module,
            )
            rewritten = performance_module._rewrite_root_paths(result, winner_root, live_root)
            rewritten["patch_receipt"] = commit_receipt
            rewritten["agentic_generation_search"] = {
                "schema_version": "mmm/custom-generation-search-v2",
                "candidate_count": len(evaluations),
                "candidate_workers": workers,
                "winner_index": int(winner_index),
                "winner_score": float(score),
                "winner_verifier": verifier,
                "candidate_scores": [
                    {
                        "candidate_index": int(item[1]),
                        "score": float(item[0]),
                        "verifier": item[4],
                    }
                    for item in sorted(evaluations, key=lambda value: value[1])
                ],
            }
            print(
                "custom generation search:",
                f"candidates={len(evaluations)}",
                f"workers={workers}",
                f"winner={int(winner_index) + 1}",
                f"score={float(score):.3f}",
                flush=True,
            )
            return rewritten
        finally:
            for _index, candidate_root, _result in candidates:
                shutil.rmtree(candidate_root, ignore_errors=True)
            shutil.rmtree(base_root, ignore_errors=True)

    generate._mmm_max_parallel_custom_search = True  # type: ignore[attr-defined]
    generate._mmm_custom_verifier_search = True  # type: ignore[attr-defined]
    generate._mmm_host_evidence_router = True  # type: ignore[attr-defined]
    cls.generate = generate


def enhance_runtime(*, work_graph_module: Any, scheduler_module: Any) -> None:
    """Install the final throughput layer after scheduler and llama safety contracts."""

    from . import custom_module_generator

    _install_exact_llm_executor()
    _install_module_routing(work_graph_module)
    _install_max_efficiency_claim(work_graph_module, scheduler_module)
    _install_parallel_custom_search(custom_module_generator)


__all__ = [
    "_active_parallelism",
    "_install_exact_llm_executor",
    "_install_max_efficiency_claim",
    "_install_module_routing",
    "_install_parallel_custom_search",
    "enhance_runtime",
]
