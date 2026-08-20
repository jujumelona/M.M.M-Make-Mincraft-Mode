from __future__ import annotations

import os
import threading
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sequence

from .project_write_lock import project_write_lock

_ORCHESTRATOR_WORKER = "mmm-orchestrator"
_RESOURCE_CAPACITIES = {
    # llama_parallel_runtime_contract replaces the LLM value with the selected native
    # slot capacity after this safety layer is installed.
    "llm": 1,
    "image_gpu": 1,
    "commit": 1,
}
_RESOURCE_PRIORITY = {"llm": 0, "image_gpu": 1, "cpu_io": 2, "commit": 3}
# These stages mutate stage-global registries, so work within the same domain is
# serialized. Different domains are independent and can execute concurrently.
_STAGE_WRITE_LOCKS = {
    "content": threading.RLock(),
    "system": threading.RLock(),
    "entity": threading.RLock(),
}
_SERIAL_CPU_STAGES = tuple(f"generate:{stage}" for stage in _STAGE_WRITE_LOCKS)
_INDEX_COMMIT_LOCK = threading.RLock()
_SHARED_LOCAL_GPU_LANE: ContextVar[bool] = ContextVar(
    "mmm_shared_local_gpu_lane",
    default=False,
)


def _cpu_capacity() -> int:
    return max(1, min(4, os.cpu_count() or 2))


def _capacities() -> dict[str, int]:
    return {
        "cpu_io": _cpu_capacity(),
        **_RESOURCE_CAPACITIES,
    }


def _resource_sql(alias: str) -> str:
    prefix = f"{alias}." if alias else ""
    return (
        "CASE COALESCE(json_extract("
        + prefix
        + "payload_json, '$.resource_class'), 'cpu_io') "
        "WHEN 'llm' THEN 'llm' "
        "WHEN 'image_gpu' THEN 'image_gpu' "
        "WHEN 'commit' THEN 'commit' "
        "ELSE 'cpu_io' END"
    )


def _orchestrator_owner(ledger: Any) -> str:
    owner = getattr(ledger, "_mmm_parallel_lease_owner", "")
    if owner:
        return str(owner)
    owner = f"{_ORCHESTRATOR_WORKER}:{os.getpid()}:{uuid.uuid4().hex}"
    setattr(ledger, "_mmm_parallel_lease_owner", owner)
    return owner


def _path_values(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    paths: list[str] = []
    for item in value:
        if isinstance(item, (str, Path)):
            rendered = str(item).strip()
            if rendered:
                paths.append(rendered)
    return tuple(paths)


def _receipt_touched_paths(receipt: Any) -> tuple[str, ...]:
    """Collect source paths from nested generator and patch receipts deterministically."""

    ordered: list[str] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        normalized = path.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            operation = value.get("operation")
            path = value.get("path")
            if operation in {"create", "replace", "edit", "delete"} and isinstance(path, str):
                add(path)

            for key in ("touched_paths", "written_files", "deleted_files", "removed_files"):
                for item in _path_values(value.get(key)):
                    add(item)

            for item in _path_values(value.get("files")):
                add(item)

            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(receipt)
    return tuple(ordered)


def _profile_uses_shared_local_gpu(profile: str, registry: Any | None = None) -> bool:
    """Return whether generation text and image roles contend for one local GPU.

    Remote/API profiles intentionally remain independent. Local llama/vLLM plus local
    diffusion profiles share one physical device. Text requests may share the resident
    model up to native slot capacity, while diffusion remains mutually exclusive with
    every text slot.
    """

    try:
        if registry is None:
            from .model_registry import ModelRegistry

            registry = ModelRegistry()
        text = registry.role(profile, "coder")
        image = registry.role(profile, "image_generator")
    except Exception:
        return False
    return (
        str(getattr(text, "provider", "")) == "local"
        and str(getattr(text, "adapter", "")) in {"llama_cpp", "vllm"}
        and bool(getattr(text, "exclusive_gpu", False))
        and str(getattr(image, "provider", "")) == "local"
        and str(getattr(image, "adapter", "")) == "image_diffusion"
        and bool(getattr(image, "exclusive_gpu", False))
    )


def _install_profile_gpu_lane(orchestrator_module: Any) -> None:
    orchestrator_cls = orchestrator_module.CompleteProductionOrchestrator
    current = orchestrator_cls._execute_generation_work
    if getattr(current, "_mmm_profile_shared_gpu_lane", False):
        return

    @wraps(current)
    def execute_generation_work(self: Any, *args: Any, **kwargs: Any):
        router = kwargs.get("router")
        registry = getattr(router, "registry", None) if router is not None else None
        profile = (
            str(getattr(router, "profile", ""))
            if router is not None
            else str(getattr(self, "profile", ""))
        )
        shared = _profile_uses_shared_local_gpu(profile, registry)
        token = _SHARED_LOCAL_GPU_LANE.set(shared)
        try:
            return current(self, *args, **kwargs)
        finally:
            _SHARED_LOCAL_GPU_LANE.reset(token)

    execute_generation_work._mmm_profile_shared_gpu_lane = True  # type: ignore[attr-defined]
    orchestrator_cls._execute_generation_work = execute_generation_work


def _install_lane_aware_claim(work_graph_module: Any) -> None:
    ledger_cls = work_graph_module.DurableWorkLedger
    current = ledger_cls.claim_ready
    if getattr(current, "_mmm_parallel_lane_claim", False):
        return

    @wraps(current)
    def claim_ready(
        self: Any,
        worker_id: str,
        *,
        stages: Sequence[str] = (),
        lease_seconds: int = 900,
    ) -> dict[str, Any] | None:
        if worker_id != _ORCHESTRATOR_WORKER:
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

        owner = _orchestrator_owner(self)
        resource_expr = _resource_sql("")
        task_resource_expr = _resource_sql("task")
        active_resource_expr = _resource_sql("active")
        stage_sql = ""
        stage_params: tuple[Any, ...] = ()
        if stages:
            placeholders = ",".join("?" for _ in stages)
            stage_sql = f" AND task.stage IN ({placeholders})"
            stage_params = tuple(stages)

        def ready_node_id(connection: Any) -> str | None:
            capacities = {
                lane: max(1, int(capacity))
                for lane, capacity in _capacities().items()
            }
            running = {
                str(resource_class): int(count)
                for resource_class, count in connection.execute(
                    f"""
                    SELECT {resource_expr}, COUNT(*)
                    FROM tasks
                    WHERE state = ? AND stage LIKE 'generate:%'
                    GROUP BY {resource_expr}
                    """,
                    (work_graph_module.WorkState.RUNNING.value,),
                )
            }
            free_lanes = {
                lane
                for lane, capacity in capacities.items()
                if running.get(lane, 0) < capacity
            }
            if _SHARED_LOCAL_GPU_LANE.get():
                # Text slots are concurrent readers of one resident local model.
                # Diffusion is the exclusive writer on that physical GPU.
                if running.get("image_gpu", 0) > 0:
                    free_lanes.discard("llm")
                if running.get("llm", 0) > 0:
                    free_lanes.discard("image_gpu")
            if not free_lanes:
                return None

            ordered_lanes = tuple(
                sorted(
                    free_lanes,
                    key=lambda lane: (_RESOURCE_PRIORITY.get(lane, 9), lane),
                )
            )
            lane_placeholders = ",".join("?" for _ in ordered_lanes)
            serial_stage_placeholders = ",".join("?" for _ in _SERIAL_CPU_STAGES)
            params = (
                work_graph_module.WorkState.PENDING.value,
                work_graph_module.WorkState.SUCCEEDED.value,
                *_SERIAL_CPU_STAGES,
                work_graph_module.WorkState.RUNNING.value,
                *stage_params,
                *ordered_lanes,
            )
            rows = connection.execute(
                f"""
                WITH ready AS (
                    SELECT task.node_id, {task_resource_expr} AS resource_class
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
                      AND NOT (
                        {task_resource_expr} = 'cpu_io'
                        AND task.stage IN ({serial_stage_placeholders})
                        AND EXISTS (
                            SELECT 1
                            FROM tasks AS active
                            WHERE active.state = ?
                              AND active.stage = task.stage
                              AND {active_resource_expr} = 'cpu_io'
                        )
                      )
                      {stage_sql}
                      AND {task_resource_expr} IN ({lane_placeholders})
                ), ranked AS (
                    SELECT node_id, resource_class,
                           ROW_NUMBER() OVER (
                               PARTITION BY resource_class ORDER BY node_id
                           ) AS resource_rank
                    FROM ready
                )
                SELECT node_id, resource_class
                FROM ranked
                WHERE resource_rank = 1
                """,
                params,
            ).fetchall()
            if not rows:
                return None

            node_id, _resource = min(
                ((str(node), str(resource)) for node, resource in rows),
                key=lambda item: (
                    running.get(item[1], 0) / capacities.get(item[1], 1),
                    _RESOURCE_PRIORITY.get(item[1], 9),
                    item[0],
                ),
            )
            return node_id

        now = time.time()
        renew_before = now + max(1.0, lease_seconds * 0.5)
        connection = self._connect()
        maintenance_due = connection.execute(
            """
            SELECT 1
            FROM tasks
            WHERE state = ? AND lease_until IS NOT NULL
              AND (
                lease_until < ?
                OR (
                  lease_owner = ?
                  AND lease_until >= ? AND lease_until <= ?
                )
              )
            LIMIT 1
            """,
            (
                work_graph_module.WorkState.RUNNING.value,
                now,
                owner,
                now,
                renew_before,
            ),
        ).fetchone()
        if maintenance_due is None and ready_node_id(connection) is None:
            return None

        with connection:
            now = time.time()
            renew_before = now + max(1.0, lease_seconds * 0.5)
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

            node_id = ready_node_id(connection)
            if node_id is None:
                return None

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
                return None
        return self.task(node_id)

    claim_ready._mmm_parallel_lane_claim = True  # type: ignore[attr-defined]
    claim_ready._mmm_exact_executor_fairness = True  # type: ignore[attr-defined]
    claim_ready._mmm_stage_lock_admission = True  # type: ignore[attr-defined]
    # The consolidated safety claimant already owns the max-efficiency semantics.
    # Any late compatibility installer must not wrap it with another transaction.
    claim_ready._mmm_max_efficiency_claim = True  # type: ignore[attr-defined]
    ledger_cls.claim_ready = claim_ready


def _stage_write_lock(node: Any) -> threading.RLock | None:
    if str(getattr(node, "resource_class", "")) != "cpu_io":
        return None
    stage = str(getattr(node, "stage", ""))
    if not stage.startswith("generate:"):
        return None
    return _STAGE_WRITE_LOCKS.get(stage.split(":", 1)[1])


def _install_index_commit_order(
    work_graph_module: Any,
    orchestrator_module: Any,
) -> None:
    orchestrator_cls = orchestrator_module.CompleteProductionOrchestrator
    current = orchestrator_cls._run_work_node
    if getattr(current, "_mmm_index_before_success", False):
        return

    @wraps(current)
    def run_work_node(
        ledger: Any,
        node: Any,
        *,
        action: Callable[[], dict[str, Any]],
        validate_cached: Callable[[dict[str, Any]], bool],
        shared_index: Any | None = None,
    ) -> dict[str, Any]:
        cached = ledger.cached_receipt(node.node_id, input_hash=node.input_hash)
        if cached is not None and validate_cached(cached):
            return cached
        if cached is not None:
            ledger.invalidate(node.node_id)

        current_task = ledger.task(node.node_id)
        if current_task["state"] in {"failed", "input_required", "cancelled"}:
            ledger.retry(node.node_id)
            current_task = ledger.task(node.node_id)

        ledger.raise_if_cancelled()
        if current_task["state"] != "running":
            ledger.begin(node.node_id, worker_id="complete-orchestrator")

        try:
            stage_lock = _stage_write_lock(node)
            if stage_lock is not None:
                with stage_lock:
                    receipt = action()
            elif (
                node.resource_class == "commit"
                and shared_index is not None
                and hasattr(shared_index, "root")
            ):
                with project_write_lock(shared_index.root):
                    receipt = action()
            else:
                receipt = action()
            if not isinstance(receipt, dict):
                raise orchestrator_module.CompleteProductionError(
                    f"Work node {node.node_id} returned a non-object receipt."
                )
            ledger.raise_if_cancelled()

            if shared_index is not None:
                touched = _receipt_touched_paths(receipt)
                if touched:
                    try:
                        with _INDEX_COMMIT_LOCK:
                            shared_index.update_files(touched)
                            shared_index.write_manifest()
                    except Exception as exc:
                        raise orchestrator_module.CompleteProductionError(
                            f"Shared ProjectIndex commit failed for {node.node_id}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

            ledger.succeed(node.node_id, receipt)
            return receipt
        except BaseException as exc:
            try:
                if ledger.task(node.node_id)["state"] == "running":
                    ledger.fail(node.node_id, f"{type(exc).__name__}: {exc}")
            except work_graph_module.WorkGraphError:
                pass
            raise

    run_work_node._mmm_index_before_success = True
    orchestrator_cls._run_work_node = staticmethod(run_work_node)


def install(
    *,
    work_graph_module: Any,
    orchestrator_module: Any,
) -> None:
    _install_profile_gpu_lane(orchestrator_module)
    _install_lane_aware_claim(work_graph_module)
    _install_index_commit_order(work_graph_module, orchestrator_module)


__all__ = [
    "_profile_uses_shared_local_gpu",
    "_receipt_touched_paths",
    "_stage_write_lock",
    "install",
]
