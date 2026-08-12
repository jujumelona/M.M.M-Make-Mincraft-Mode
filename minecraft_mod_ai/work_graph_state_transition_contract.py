from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Any


def install(work_graph_module: Any) -> None:
    """Make durable task/checkpoint transitions row-local and monotonic.

    MMM now reuses one SQLite connection per ledger/thread. ``Connection.total_changes``
    is cumulative for that connection, so it cannot prove that the current UPDATE
    matched a row. The legacy failure paths also allowed a late worker to overwrite a
    cancelled/succeeded state. Use statement ``rowcount`` and RUNNING-state predicates
    so stale completions fail closed instead of rewriting newer durable state.
    """

    cls = work_graph_module.DurableWorkLedger

    current_fail = cls.fail
    if not getattr(current_fail, "_mmm_fenced_transition", False):

        @wraps(current_fail)
        def fail(
            self: Any,
            node_id: str,
            error: str,
            *,
            input_required: bool = False,
        ) -> dict[str, Any]:
            target = (
                work_graph_module.WorkState.INPUT_REQUIRED
                if input_required
                else work_graph_module.WorkState.FAILED
            )
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT state FROM tasks WHERE node_id = ?",
                    (node_id,),
                ).fetchone()
                if row is None:
                    raise work_graph_module.WorkGraphError(
                        f"Unknown work node: {node_id}"
                    )
                if row[0] == work_graph_module.WorkState.CANCELLED.value:
                    connection.commit()
                    return self.task(node_id)
                if row[0] != work_graph_module.WorkState.RUNNING.value:
                    raise work_graph_module.WorkGraphError(
                        f"Work node {node_id} cannot fail from state {row[0]}."
                    )
                cursor = connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, error = ?, lease_owner = NULL,
                        lease_until = NULL, updated_at = ?
                    WHERE node_id = ? AND state = ?
                    """,
                    (
                        target.value,
                        str(error)[:16_384],
                        time.time(),
                        node_id,
                        work_graph_module.WorkState.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Work node changed while failing: {node_id}"
                    )
                connection.commit()
            return self.task(node_id)

        fail._mmm_fenced_transition = True  # type: ignore[attr-defined]
        cls.fail = fail

    current_succeed_checkpoint = cls.succeed_checkpoint
    if not getattr(current_succeed_checkpoint, "_mmm_fenced_transition", False):

        @wraps(current_succeed_checkpoint)
        def succeed_checkpoint(
            self: Any,
            checkpoint_id: str,
            *,
            input_hash: str,
            receipt: dict[str, Any],
        ) -> None:
            rendered = work_graph_module.canonical_json(receipt)
            output_hash = "sha256:" + hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest()
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE checkpoints
                    SET state = ?, receipt_json = ?, output_hash = ?,
                        error = NULL, updated_at = ?
                    WHERE checkpoint_id = ? AND input_hash = ? AND state = ?
                    """,
                    (
                        work_graph_module.WorkState.SUCCEEDED.value,
                        rendered,
                        output_hash,
                        time.time(),
                        checkpoint_id,
                        input_hash,
                        work_graph_module.WorkState.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint changed while running: {checkpoint_id}"
                    )
                connection.commit()

        succeed_checkpoint._mmm_fenced_transition = True  # type: ignore[attr-defined]
        cls.succeed_checkpoint = succeed_checkpoint

    current_fail_checkpoint = cls.fail_checkpoint
    if not getattr(current_fail_checkpoint, "_mmm_fenced_transition", False):

        @wraps(current_fail_checkpoint)
        def fail_checkpoint(
            self: Any,
            checkpoint_id: str,
            *,
            input_hash: str,
            error: str,
        ) -> None:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT input_hash, state FROM checkpoints
                    WHERE checkpoint_id = ?
                    """,
                    (checkpoint_id,),
                ).fetchone()
                if row is None:
                    raise work_graph_module.WorkGraphError(
                        f"Unknown checkpoint: {checkpoint_id}"
                    )
                if row[0] != input_hash:
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint input changed while running: {checkpoint_id}"
                    )
                if row[1] == work_graph_module.WorkState.CANCELLED.value:
                    connection.commit()
                    return
                if row[1] != work_graph_module.WorkState.RUNNING.value:
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint {checkpoint_id} cannot fail from state {row[1]}."
                    )
                cursor = connection.execute(
                    """
                    UPDATE checkpoints
                    SET state = ?, error = ?, updated_at = ?
                    WHERE checkpoint_id = ? AND input_hash = ? AND state = ?
                    """,
                    (
                        work_graph_module.WorkState.FAILED.value,
                        str(error)[:16_384],
                        time.time(),
                        checkpoint_id,
                        input_hash,
                        work_graph_module.WorkState.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint changed while failing: {checkpoint_id}"
                    )
                connection.commit()

        fail_checkpoint._mmm_fenced_transition = True  # type: ignore[attr-defined]
        cls.fail_checkpoint = fail_checkpoint


__all__ = ["install"]
