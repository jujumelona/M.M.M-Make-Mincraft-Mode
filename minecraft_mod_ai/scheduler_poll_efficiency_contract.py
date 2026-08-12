from __future__ import annotations

import threading
import time
from functools import wraps
from typing import Any


_POLL_CACHE_SECONDS = 0.04


def install(work_graph_module: Any) -> None:
    """Batch read-only generation-task polling without changing claim authority.

    The orchestrator asks ``ledger.task()`` for every generation node on each scheduler
    pass. With thousands of nodes that becomes thousands of SQLite queries every ~50ms.
    Only the orchestrator main thread needs that full-DAG polling view; worker threads
    keep exact point reads. Claims, dependency checks, leases and mutations still use
    the existing transactional database paths, so this cache can delay observation by
    at most a few milliseconds but can never authorize stale work.
    """

    ledger_cls = work_graph_module.DurableWorkLedger
    current = ledger_cls.task
    if getattr(current, "_mmm_batched_generation_poll", False):
        return

    def _snapshot(self: Any) -> dict[str, dict[str, Any]]:
        now = time.monotonic()
        cache = getattr(self, "_mmm_generation_poll_snapshot", None)
        stamp = float(getattr(self, "_mmm_generation_poll_stamp", 0.0))
        if isinstance(cache, dict) and now - stamp <= _POLL_CACHE_SECONDS:
            return cache

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT node_id, stage, input_hash, payload_json, state,
                       attempt, lease_owner, lease_until, output_hash,
                       receipt_json, error, updated_at
                FROM tasks
                WHERE stage LIKE 'generate:%'
                ORDER BY node_id
                """
            ).fetchall()
            dependencies: dict[str, list[str]] = {}
            for node_id, dependency_id in connection.execute(
                """
                SELECT edge.node_id, edge.dependency_id
                FROM edges AS edge
                JOIN tasks AS task ON task.node_id = edge.node_id
                WHERE task.stage LIKE 'generate:%'
                ORDER BY edge.node_id, edge.dependency_id
                """
            ):
                dependencies.setdefault(str(node_id), []).append(str(dependency_id))

        rendered: dict[str, dict[str, Any]] = {}
        for row in rows:
            node_id = str(row[0])
            rendered[node_id] = {
                "node_id": node_id,
                "stage": row[1],
                "input_hash": row[2],
                "payload": work_graph_module.json.loads(row[3]),
                "state": row[4],
                "attempt": row[5],
                "lease_owner": row[6],
                "lease_until": row[7],
                "output_hash": row[8],
                "receipt": (
                    work_graph_module.json.loads(row[9]) if row[9] else None
                ),
                "error": row[10],
                "updated_at": row[11],
                "dependencies": dependencies.get(node_id, []),
            }
        self._mmm_generation_poll_snapshot = rendered
        self._mmm_generation_poll_stamp = now
        return rendered

    @wraps(current)
    def task(self: Any, node_id: str) -> dict[str, Any]:
        # Worker actions and non-generation control nodes require exact point reads.
        # The main orchestrator poll is the only high-frequency full-DAG scan.
        if (
            threading.current_thread() is threading.main_thread()
            and str(node_id).startswith("generate-")
        ):
            value = _snapshot(self).get(str(node_id))
            if value is not None:
                return dict(value)
        return current(self, node_id)

    task._mmm_batched_generation_poll = True  # type: ignore[attr-defined]
    task.__wrapped__ = current  # type: ignore[attr-defined]
    ledger_cls.task = task


__all__ = ["install"]
