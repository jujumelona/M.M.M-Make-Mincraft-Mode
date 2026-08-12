from __future__ import annotations

import json
import os
import time
from collections import Counter
from typing import Any, Sequence


def _resource_class(payload_json: str) -> str:
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return "cpu_io"
    value = str(payload.get("resource_class", "cpu_io")) if isinstance(payload, dict) else "cpu_io"
    return value if value in {"cpu_io", "llm", "image_gpu", "commit"} else "cpu_io"


def install(work_graph_module: Any) -> None:
    """Claim ready DAG nodes according to the real local executor capacities.

    The orchestrator owns four CPU workers and one worker for each scarce LLM, image
    and commit lane.  Claiming more work than a lane can execute turns leases into a
    hidden queue and can leave other independent lanes idle.  This selector keeps the
    durable lease semantics but balances claims across the actual executor slots.
    """

    cls = work_graph_module.DurableWorkLedger
    current = cls.claim_ready
    if getattr(current, "_mmm_exact_executor_fairness", False):
        return

    def claim_ready_fair(
        self: Any,
        worker_id: str,
        *,
        stages: Sequence[str] = (),
        lease_seconds: int = 900,
    ) -> dict[str, Any] | None:
        if not worker_id.strip():
            raise work_graph_module.WorkGraphError("worker_id must not be empty.")
        if lease_seconds < 1:
            raise work_graph_module.WorkGraphError("lease_seconds must be positive.")

        now = time.time()
        capacities = {
            "cpu_io": min(4, os.cpu_count() or 2),
            "llm": 1,
            "image_gpu": 1,
            "commit": 1,
        }
        priority = {"llm": 0, "image_gpu": 1, "cpu_io": 2, "commit": 3}

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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

            running: Counter[str] = Counter()
            for (payload_json,) in connection.execute(
                "SELECT payload_json FROM tasks WHERE state = ?",
                (work_graph_module.WorkState.RUNNING.value,),
            ):
                running[_resource_class(str(payload_json))] += 1

            stage_sql = ""
            params: list[Any] = [
                work_graph_module.WorkState.PENDING.value,
                work_graph_module.WorkState.SUCCEEDED.value,
            ]
            if stages:
                placeholders = ",".join("?" for _ in stages)
                stage_sql = f" AND task.stage IN ({placeholders})"
                params.extend(stages)
            rows = connection.execute(
                f"""
                SELECT task.node_id, task.payload_json
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
                ORDER BY task.node_id
                LIMIT 256
                """,
                tuple(params),
            ).fetchall()

            candidates: list[tuple[float, int, str, str]] = []
            for node_id, payload_json in rows:
                resource = _resource_class(str(payload_json))
                capacity = capacities[resource]
                active = running[resource]
                if active >= capacity:
                    continue
                candidates.append(
                    (
                        active / max(1, capacity),
                        priority[resource],
                        str(node_id),
                        resource,
                    )
                )

            if not candidates:
                connection.commit()
                return None
            candidates.sort()
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
                    worker_id,
                    now + lease_seconds,
                    now,
                    node_id,
                    work_graph_module.WorkState.PENDING.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
        return self.task(node_id)

    claim_ready_fair._mmm_exact_executor_fairness = True  # type: ignore[attr-defined]
    claim_ready_fair.__wrapped__ = current  # type: ignore[attr-defined]
    cls.claim_ready = claim_ready_fair


__all__ = ["install"]
