from __future__ import annotations

import hashlib
import sys
import time
from functools import wraps
from typing import Any, Callable, TypeVar


_T = TypeVar("_T")


def install(work_graph_module: Any) -> None:
    """Make durable task/checkpoint transitions row-local and monotonic.

    Ordinary worker failure/success remains fenced to RUNNING rows. INPUT_REQUIRED is
    different: it is a control-plane blocked state and may be recorded directly from a
    pending node whose dependencies cannot yet run. Named checkpoints also get one
    explicit invalid-cache recovery path; direct begin_checkpoint() still refuses to
    restart an already successful checkpoint.
    """

    cls = work_graph_module.DurableWorkLedger

    current_succeed = cls.succeed
    if not getattr(current_succeed, "_mmm_fenced_transition", False):

        @wraps(current_succeed)
        def succeed(
            self: Any,
            node_id: str,
            receipt: dict[str, Any],
            *,
            output_hash: str = "",
        ) -> dict[str, Any]:
            rendered = work_graph_module.canonical_json(receipt)
            digest = output_hash or "sha256:" + hashlib.sha256(
                rendered.encode("utf-8")
            ).hexdigest()
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT state, output_hash, receipt_json FROM tasks
                    WHERE node_id = ?
                    """,
                    (node_id,),
                ).fetchone()
                if row is None:
                    raise work_graph_module.WorkGraphError(
                        f"Unknown work node: {node_id}"
                    )
                if row[0] == work_graph_module.WorkState.SUCCEEDED.value:
                    if row[1] == digest and row[2] == rendered:
                        connection.commit()
                        return self.task(node_id)
                    raise work_graph_module.WorkGraphError(
                        f"Work node {node_id} already succeeded with a different receipt."
                    )
                if row[0] != work_graph_module.WorkState.RUNNING.value:
                    raise work_graph_module.WorkGraphError(
                        f"Work node {node_id} cannot succeed from state {row[0]}."
                    )
                cursor = connection.execute(
                    """
                    UPDATE tasks
                    SET state = ?, output_hash = ?, receipt_json = ?,
                        lease_owner = NULL, lease_until = NULL, error = NULL,
                        updated_at = ?
                    WHERE node_id = ? AND state = ?
                    """,
                    (
                        work_graph_module.WorkState.SUCCEEDED.value,
                        digest,
                        rendered,
                        time.time(),
                        node_id,
                        work_graph_module.WorkState.RUNNING.value,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Work node changed while succeeding: {node_id}"
                    )
                connection.commit()
            return self.task(node_id)

        succeed._mmm_fenced_transition = True  # type: ignore[attr-defined]
        cls.succeed = succeed

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
                source_state = str(row[0])
                if source_state == work_graph_module.WorkState.CANCELLED.value:
                    connection.commit()
                    return self.task(node_id)
                if (
                    input_required
                    and source_state == work_graph_module.WorkState.INPUT_REQUIRED.value
                ):
                    connection.commit()
                    return self.task(node_id)
                allowed = {work_graph_module.WorkState.RUNNING.value}
                if input_required:
                    # A node can be known to require external evidence/input before its
                    # execution dependencies are satisfiable. This is not a worker
                    # failure, so forcing begin() first would be both false and often
                    # impossible (quality nodes can depend on runtime/build gates).
                    allowed.add(work_graph_module.WorkState.PENDING.value)
                if source_state not in allowed:
                    raise work_graph_module.WorkGraphError(
                        f"Work node {node_id} cannot fail from state {source_state}."
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
                        source_state,
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

    current_begin_checkpoint = cls.begin_checkpoint
    if not getattr(current_begin_checkpoint, "_mmm_fenced_transition", False):

        @wraps(current_begin_checkpoint)
        def begin_checkpoint(
            self: Any,
            checkpoint_id: str,
            *,
            stage: str,
            input_hash: str,
        ) -> None:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT input_hash, state FROM checkpoints
                    WHERE checkpoint_id = ?
                    """,
                    (checkpoint_id,),
                ).fetchone()
                if row is None:
                    connection.execute(
                        """
                        INSERT INTO checkpoints(
                            checkpoint_id, stage, input_hash, state, attempt, updated_at
                        ) VALUES (?, ?, ?, ?, 1, ?)
                        """,
                        (
                            checkpoint_id,
                            stage,
                            input_hash,
                            work_graph_module.WorkState.RUNNING.value,
                            time.time(),
                        ),
                    )
                    connection.commit()
                    return

                old_hash, old_state = str(row[0]), str(row[1])
                if old_hash == input_hash:
                    if old_state == work_graph_module.WorkState.SUCCEEDED.value:
                        connection.rollback()
                        raise work_graph_module.WorkGraphError(
                            f"Checkpoint already succeeded for this input: {checkpoint_id}"
                        )
                    if old_state == work_graph_module.WorkState.RUNNING.value:
                        connection.rollback()
                        raise work_graph_module.WorkGraphError(
                            f"Checkpoint is already running: {checkpoint_id}"
                        )
                    if old_state == work_graph_module.WorkState.CANCELLED.value:
                        connection.rollback()
                        raise work_graph_module.WorkGraphError(
                            f"Checkpoint is cancelled and requires explicit retry: {checkpoint_id}"
                        )
                elif old_state == work_graph_module.WorkState.RUNNING.value:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint input changed while another attempt is running: {checkpoint_id}"
                    )

                cursor = connection.execute(
                    """
                    UPDATE checkpoints
                    SET stage = ?, input_hash = ?, state = ?,
                        attempt = attempt + 1, receipt_json = NULL,
                        output_hash = NULL, error = NULL, updated_at = ?
                    WHERE checkpoint_id = ? AND state = ? AND input_hash = ?
                    """,
                    (
                        stage,
                        input_hash,
                        work_graph_module.WorkState.RUNNING.value,
                        time.time(),
                        checkpoint_id,
                        old_state,
                        old_hash,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint changed while starting: {checkpoint_id}"
                    )
                connection.commit()

        begin_checkpoint._mmm_fenced_transition = True  # type: ignore[attr-defined]
        cls.begin_checkpoint = begin_checkpoint

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

    if not hasattr(cls, "invalidate_checkpoint"):

        def invalidate_checkpoint(
            self: Any,
            checkpoint_id: str,
            *,
            input_hash: str,
            reason: str = "cached checkpoint output failed validation",
        ) -> bool:
            """Invalidate exactly one successful checkpoint/input pair atomically."""

            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT input_hash, state FROM checkpoints
                    WHERE checkpoint_id = ?
                    """,
                    (checkpoint_id,),
                ).fetchone()
                if row is None:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Unknown checkpoint: {checkpoint_id}"
                    )
                if str(row[0]) != input_hash:
                    connection.rollback()
                    return False
                if str(row[1]) != work_graph_module.WorkState.SUCCEEDED.value:
                    connection.commit()
                    return False
                cursor = connection.execute(
                    """
                    UPDATE checkpoints
                    SET state = ?, receipt_json = NULL, output_hash = NULL,
                        error = ?, updated_at = ?
                    WHERE checkpoint_id = ? AND input_hash = ? AND state = ?
                    """,
                    (
                        work_graph_module.WorkState.FAILED.value,
                        reason[:16_384],
                        time.time(),
                        checkpoint_id,
                        input_hash,
                        work_graph_module.WorkState.SUCCEEDED.value,
                    ),
                )
                if cursor.rowcount != 1:
                    connection.rollback()
                    raise work_graph_module.WorkGraphError(
                        f"Checkpoint changed while invalidating: {checkpoint_id}"
                    )
                connection.commit()
                return True

        cls.invalidate_checkpoint = invalidate_checkpoint

    current_named = work_graph_module.run_named_checkpoint
    if not getattr(current_named, "_mmm_validated_cache_recovery", False):

        @wraps(current_named)
        def run_named_checkpoint(
            ledger: Any,
            checkpoint_id: str,
            *,
            stage: str,
            input_value: Any,
            action: Callable[[], _T],
            encode: Callable[[_T], dict[str, Any]],
            decode: Callable[[dict[str, Any]], _T],
            validate_cached: Callable[[_T], bool] | None = None,
        ) -> _T:
            ledger.raise_if_cancelled()
            input_hash = work_graph_module._hash_json(input_value)
            cached = ledger.cached_checkpoint(
                checkpoint_id,
                input_hash=input_hash,
            )
            if cached is not None:
                decoded = decode(cached)
                if validate_cached is None or validate_cached(decoded):
                    return decoded
                ledger.invalidate_checkpoint(
                    checkpoint_id,
                    input_hash=input_hash,
                )
            ledger.begin_checkpoint(
                checkpoint_id,
                stage=stage,
                input_hash=input_hash,
            )
            try:
                value = action()
                ledger.succeed_checkpoint(
                    checkpoint_id,
                    input_hash=input_hash,
                    receipt=encode(value),
                )
                return value
            except Exception as exc:
                ledger.fail_checkpoint(
                    checkpoint_id,
                    input_hash=input_hash,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise

        run_named_checkpoint._mmm_validated_cache_recovery = True  # type: ignore[attr-defined]
        work_graph_module.run_named_checkpoint = run_named_checkpoint

        # complete_orchestrator imports this helper by value before architecture
        # contracts are installed. Rebind only that already-imported exact symbol so
        # the runtime and direct work_graph callers share one recovery semantics.
        orchestrator = sys.modules.get(f"{work_graph_module.__package__}.complete_orchestrator")
        if (
            orchestrator is not None
            and getattr(orchestrator, "run_named_checkpoint", None) is current_named
        ):
            orchestrator.run_named_checkpoint = run_named_checkpoint


__all__ = ["install"]
