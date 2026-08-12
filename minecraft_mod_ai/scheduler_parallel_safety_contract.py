from __future__ import annotations

import os
import threading
import time
import uuid
from functools import wraps
from typing import Any, Callable, Sequence

_ORCHESTRATOR_WORKER = "mmm-orchestrator"
_RESOURCE_CAPACITIES = {
    "llm": 1,
    "image_gpu": 1,
    "commit": 1,
}
_INDEX_COMMIT_LOCK = threading.RLock()


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
    owner = (
        f"{_ORCHESTRATOR_WORKER}:{os.getpid()}:"
        f"{uuid.uuid4().hex}"
    )
    setattr(ledger, "_mmm_parallel_lease_owner", owner)
    return owner


def _install_thread_local_connections(work_graph_module: Any) -> None:
    """Bound SQLite connections to one reusable handle per ledger/thread.

    DurableWorkLedger historically created a fresh sqlite3.Connection for every
    ``with self._connect()`` call. sqlite3.Connection.__exit__ commits or rolls
    back but does not close the handle, so a 50 ms scheduler poll over a large
    DAG can create an unbounded stream of connections while also repeating the
    WAL/pragma setup.  The scheduler already uses a bounded thread topology, so
    a thread-local connection gives both correct sqlite thread affinity and a
    fixed connection count.
    """

    ledger_cls = work_graph_module.DurableWorkLedger
    current = ledger_cls._connect
    if getattr(current, "_mmm_thread_local_connection", False):
        return

    @wraps(current)
    def connect(self: Any):
        local = getattr(self, "_mmm_sqlite_local", None)
        if local is None:
            local = threading.local()
            setattr(self, "_mmm_sqlite_local", local)

        connection = getattr(local, "connection", None)
        pid = getattr(local, "pid", None)
        if connection is not None and pid != os.getpid():
            # A connection must never be inherited across a fork.  Closing can
            # fail for an already-invalid inherited handle; either way replace it.
            try:
                connection.close()
            except Exception:
                pass
            connection = None

        if connection is None:
            connection = work_graph_module.sqlite3.connect(
                self.path,
                timeout=30,
            )
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            local.connection = connection
            local.pid = os.getpid()
        return connection

    connect._mmm_thread_local_connection = True
    ledger_cls._connect = connect


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
            raise work_graph_module.WorkGraphError(
                "worker_id must not be empty."
            )
        if lease_seconds < 1:
            raise work_graph_module.WorkGraphError(
                "lease_seconds must be positive."
            )

        now = time.time()
        owner = _orchestrator_owner(self)
        capacities = _capacities()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")

            # The scheduler polls claim_ready while futures are alive. Renew only
            # leases owned by this exact process/run token, so a crashed process
            # cannot be revived accidentally by a later run that uses the same
            # public worker label.
            connection.execute(
                """
                UPDATE tasks
                SET lease_until = ?, updated_at = ?
                WHERE state = ? AND lease_owner = ?
                  AND lease_until IS NOT NULL AND lease_until >= ?
                """,
                (
                    now + lease_seconds,
                    now,
                    work_graph_module.WorkState.RUNNING.value,
                    owner,
                    now,
                ),
            )

            # Preserve the durable-ledger recovery rule for genuinely abandoned
            # work. Expired work is reclaimable by this or another worker.
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

            resource_expr = _resource_sql("")
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
            free_lanes = tuple(
                lane
                for lane, capacity in capacities.items()
                if running.get(lane, 0) < capacity
            )
            if not free_lanes:
                connection.commit()
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

            lane_placeholders = ",".join("?" for _ in free_lanes)
            params.extend(free_lanes)
            task_resource_expr = _resource_sql("task")
            row = connection.execute(
                f"""
                SELECT task.node_id
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
                  AND {task_resource_expr} IN ({lane_placeholders})
                ORDER BY task.node_id
                LIMIT 1
                """,
                tuple(params),
            ).fetchone()
            if row is None:
                connection.commit()
                return None

            node_id = str(row[0])
            connection.execute(
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
            connection.commit()
        return self.task(node_id)

    claim_ready._mmm_parallel_lane_claim = True
    ledger_cls.claim_ready = claim_ready


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
        cached = ledger.cached_receipt(
            node.node_id,
            input_hash=node.input_hash,
        )
        if cached is not None and validate_cached(cached):
            return cached
        if cached is not None:
            ledger.invalidate(node.node_id)

        current_task = ledger.task(node.node_id)
        if current_task["state"] in {
            "failed",
            "input_required",
            "cancelled",
        }:
            ledger.retry(node.node_id)
            current_task = ledger.task(node.node_id)

        ledger.raise_if_cancelled()
        if current_task["state"] != "running":
            ledger.begin(
                node.node_id,
                worker_id="complete-orchestrator",
            )

        try:
            receipt = action()
            if not isinstance(receipt, dict):
                raise orchestrator_module.CompleteProductionError(
                    f"Work node {node.node_id} returned a non-object receipt."
                )
            ledger.raise_if_cancelled()

            # Dependency readiness is driven by ledger SUCCEEDED. Therefore every
            # shared index mutation that a dependent node may read must become
            # visible before that state transition. Serialize update+manifest as
            # one critical section to avoid lost updates between CPU workers.
            if shared_index is not None:
                touched = (
                    receipt.get("touched_paths")
                    or receipt.get("written_files")
                    or []
                )
                if touched:
                    try:
                        with _INDEX_COMMIT_LOCK:
                            shared_index.update_files(touched)
                            shared_index.write_manifest()
                    except Exception:
                        # Preserve the existing best-effort index behavior. The
                        # important ordering guarantee is that this attempt is
                        # complete before dependency success becomes observable.
                        pass

            ledger.succeed(node.node_id, receipt)
            return receipt
        except BaseException as exc:
            try:
                if ledger.task(node.node_id)["state"] == "running":
                    ledger.fail(
                        node.node_id,
                        f"{type(exc).__name__}: {exc}",
                    )
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
    _install_thread_local_connections(work_graph_module)
    _install_lane_aware_claim(work_graph_module)
    _install_index_commit_order(
        work_graph_module,
        orchestrator_module,
    )
