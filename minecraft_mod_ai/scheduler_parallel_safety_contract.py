from __future__ import annotations

import os
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sequence

from .project_write_lock import project_write_lock

_ORCHESTRATOR_WORKER = "mmm-orchestrator"
_RESOURCE_CAPACITIES = {
    "llm": 1,
    "image_gpu": 1,
    "commit": 1,
}
# These stages mutate some stage-global registries, so work within the same stage is
# serialized. They no longer share one global commit lane: content, systems and
# entities can execute concurrently because their shared files are disjoint domains.
_STAGE_WRITE_LOCKS = {
    "content": threading.RLock(),
    "system": threading.RLock(),
    "entity": threading.RLock(),
}
_INDEX_COMMIT_LOCK = threading.RLock()


def _cpu_capacity() -> int:
    return max(1, min(4, os.cpu_count() or 2))


def _pipeline_shard_size(name: str, default: int, upper: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(max(1, upper), value))


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


def _content_node_is_cpu_safe(payload: dict[str, Any]) -> bool:
    """Keep unknown integration modules out of the CPU lane.

    Normal extended content and research shards are deterministic. The built-in local
    AI sidecar integration is deterministic too. Other integration modules can fall
    back to CustomModuleGenerator and therefore must not bypass the one-request LLM
    lane merely because their topological stage is named ``content``.
    """

    members = payload.get("members")
    if not isinstance(members, list):
        return False
    for member in members:
        if not isinstance(member, dict):
            return False
        if str(member.get("kind", "")) != "integration":
            continue
        config = member.get("config")
        if not isinstance(config, dict):
            return False
        if str(config.get("integration_type", "")) != "mmm_local_ai_sidecar":
            return False
    return True


def _install_pipeline_shards(work_graph_module: Any) -> None:
    """Expose dependency progress instead of hiding dozens of modules in one node.

    Custom modules are already generated one-by-one inside a shard, so a 48-member
    custom shard does not reduce LLM calls; it only prevents downstream nodes from
    starting until all 48 serial calls finish. One custom module per DAG node releases
    dependencies immediately while the single LLM lane continues with the next one.

    Entity generation is deterministic but each node also performs post-generation
    review. Small entity shards let the next entity generation overlap the prior
    node's review without concurrently mutating the same entity registry.
    """

    current = work_graph_module._module_shards
    if getattr(current, "_mmm_pipeline_granularity", False):
        return

    @wraps(current)
    def module_shards(modules: Any, *, policy: Any):
        for stage, members in current(modules, policy=policy):
            if stage == "custom":
                size = _pipeline_shard_size(
                    "MMM_CUSTOM_PIPELINE_SHARD_SIZE",
                    1,
                    len(members),
                )
            elif stage == "entity":
                size = _pipeline_shard_size(
                    "MMM_ENTITY_PIPELINE_SHARD_SIZE",
                    2,
                    len(members),
                )
            else:
                yield stage, members
                continue
            for offset in range(0, len(members), size):
                yield stage, tuple(members[offset : offset + size])

    module_shards._mmm_pipeline_granularity = True  # type: ignore[attr-defined]
    work_graph_module._module_shards = module_shards


def _install_generation_lanes(work_graph_module: Any) -> None:
    """Use CPU workers for independent deterministic stages, not one global commit queue."""

    current = work_graph_module._node
    if getattr(current, "_mmm_stage_parallel_generation_lanes", False):
        return

    @wraps(current)
    def node(
        node_id: str,
        stage: str,
        dependencies: Any,
        payload: dict[str, Any],
    ):
        normalized = dict(payload)
        if "resource_class" not in normalized and normalized.get("kind") == "module-shard":
            generation_stage = str(normalized.get("generation_stage", ""))
            if generation_stage in {"system", "entity"}:
                normalized["resource_class"] = "cpu_io"
            elif generation_stage == "content" and _content_node_is_cpu_safe(normalized):
                normalized["resource_class"] = "cpu_io"
            elif generation_stage == "audio-binding":
                normalized["resource_class"] = "commit"
        return current(node_id, stage, dependencies, normalized)

    node._mmm_stage_parallel_generation_lanes = True  # type: ignore[attr-defined]
    # Keep the legacy marker because bootstrap/tests may inspect it across live reloads.
    node._mmm_shared_write_commit_lane = True  # type: ignore[attr-defined]
    work_graph_module._node = node


def _install_thread_local_connections(work_graph_module: Any) -> None:
    """Bound SQLite connections to one reusable handle per ledger/thread."""

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
            raise work_graph_module.WorkGraphError("worker_id must not be empty.")
        if lease_seconds < 1:
            raise work_graph_module.WorkGraphError("lease_seconds must be positive.")

        now = time.time()
        owner = _orchestrator_owner(self)
        capacities = _capacities()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
                # Same-stage generators can share manifests/initializers, so serialize
                # only that domain. Different domains remain concurrent.
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
    _install_pipeline_shards(work_graph_module)
    _install_generation_lanes(work_graph_module)
    _install_thread_local_connections(work_graph_module)
    _install_lane_aware_claim(work_graph_module)
    _install_index_commit_order(work_graph_module, orchestrator_module)


__all__ = [
    "_content_node_is_cpu_safe",
    "_receipt_touched_paths",
    "_stage_write_lock",
    "install",
]
