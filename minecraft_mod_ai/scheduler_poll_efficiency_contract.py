from __future__ import annotations

import threading
from functools import wraps
from typing import Any


def install(work_graph_module: Any) -> None:
    """Batch one full generation-state scan without carrying stale state forward.

    ``_execute_generation_work`` reads every generation node once per scheduler pass.
    Thousands of point reads are wasteful, but a time-based cache can survive a worker
    completion and create a false deadlock.  This contract therefore gives the main
    scheduler thread exactly one snapshot budget equal to the number of generation
    tasks. After that many reads the snapshot is discarded. ``claim_ready`` also clears
    it before and after every transactional claim. Worker threads always use exact point
    reads, and database transactions remain the only authority for readiness/leases.
    """

    ledger_cls = work_graph_module.DurableWorkLedger

    def _clear(self: Any) -> None:
        setattr(self, "_mmm_generation_poll_snapshot", None)
        setattr(self, "_mmm_generation_poll_reads_left", 0)

    current_task = ledger_cls.task
    if not getattr(current_task, "_mmm_batched_generation_poll", False):

        def _snapshot(self: Any) -> dict[str, dict[str, Any]]:
            cache = getattr(self, "_mmm_generation_poll_snapshot", None)
            reads_left = int(getattr(self, "_mmm_generation_poll_reads_left", 0))
            if isinstance(cache, dict) and reads_left > 0:
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
                    dependencies.setdefault(str(node_id), []).append(
                        str(dependency_id)
                    )

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
                        work_graph_module.json.loads(row[9])
                        if row[9]
                        else None
                    ),
                    "error": row[10],
                    "updated_at": row[11],
                    "dependencies": dependencies.get(node_id, []),
                }
            self._mmm_generation_poll_snapshot = rendered
            self._mmm_generation_poll_reads_left = len(rendered)
            return rendered

        @wraps(current_task)
        def task(self: Any, node_id: str) -> dict[str, Any]:
            if (
                threading.current_thread() is threading.main_thread()
                and str(node_id).startswith("generate-")
            ):
                snapshot = _snapshot(self)
                value = snapshot.get(str(node_id))
                reads_left = max(
                    0,
                    int(getattr(self, "_mmm_generation_poll_reads_left", 0)) - 1,
                )
                self._mmm_generation_poll_reads_left = reads_left
                if reads_left == 0:
                    # Never carry a scheduler scan across iterations.
                    self._mmm_generation_poll_snapshot = None
                if value is not None:
                    return dict(value)
            return current_task(self, node_id)

        task._mmm_batched_generation_poll = True  # type: ignore[attr-defined]
        task.__wrapped__ = current_task  # type: ignore[attr-defined]
        ledger_cls.task = task

    # This installer is intentionally safe to call again after late scheduler safety
    # contracts replace claim_ready. Keep the task wrapper idempotent while re-binding
    # cache invalidation around whichever transactional claim implementation is final.
    current_claim = ledger_cls.claim_ready
    if not getattr(current_claim, "_mmm_poll_snapshot_fence", False):

        @wraps(current_claim)
        def claim_ready(self: Any, *args: Any, **kwargs: Any):
            _clear(self)
            try:
                return current_claim(self, *args, **kwargs)
            finally:
                _clear(self)

        claim_ready._mmm_poll_snapshot_fence = True  # type: ignore[attr-defined]
        claim_ready.__wrapped__ = current_claim  # type: ignore[attr-defined]
        ledger_cls.claim_ready = claim_ready


__all__ = ["install"]
