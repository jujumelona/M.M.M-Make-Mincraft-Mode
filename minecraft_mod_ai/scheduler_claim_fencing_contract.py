from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from .project_write_lock import project_write_lock

_INDEX_COMMIT_LOCK = threading.RLock()


def _snapshot_claim(work_graph_module: Any, ledger: Any, node_id: str) -> tuple[int, str]:
    current = ledger.task(node_id)
    if current["state"] != work_graph_module.WorkState.RUNNING.value:
        raise work_graph_module.WorkGraphError(
            f"Work node {node_id} is not running while capturing its claim: "
            f"{current['state']}"
        )
    attempt = current.get("attempt")
    owner = current.get("lease_owner")
    if type(attempt) is not int or attempt < 1:
        raise work_graph_module.WorkGraphError(
            f"Work node {node_id} has an invalid running attempt."
        )
    if not isinstance(owner, str) or not owner:
        raise work_graph_module.WorkGraphError(
            f"Work node {node_id} has no running owner."
        )
    return attempt, owner


def _fenced_fail(
    work_graph_module: Any,
    ledger: Any,
    node_id: str,
    *,
    attempt: int,
    owner: str,
    error: BaseException,
) -> None:
    with ledger._connect() as connection:
        cursor = connection.execute(
            """
            UPDATE tasks
            SET state = ?, error = ?, lease_owner = NULL,
                lease_until = NULL, updated_at = ?
            WHERE node_id = ? AND state = ? AND attempt = ? AND lease_owner = ?
            """,
            (
                work_graph_module.WorkState.FAILED.value,
                f"{type(error).__name__}: {error}"[:16_384],
                time.time(),
                node_id,
                work_graph_module.WorkState.RUNNING.value,
                attempt,
                owner,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return
        connection.commit()


def _commit_success(
    work_graph_module: Any,
    orchestrator_module: Any,
    ledger: Any,
    node_id: str,
    receipt: dict[str, Any],
    *,
    attempt: int,
    owner: str,
    shared_index: Any | None,
) -> None:
    rendered = work_graph_module.canonical_json(receipt)
    digest = "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    # Keep the claim fence live while publishing the shared index and durable success.
    # This closes the window where a reclaimed attempt could become current between an
    # index check and the task-state commit. Persist the independent receipt hash in the
    # same transaction; otherwise the receipt-integrity owner must invalidate the
    # freshly committed success on the very next read.
    with ledger._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT state, attempt, lease_owner FROM tasks WHERE node_id = ?
            """,
            (node_id,),
        ).fetchone()
        expected = (
            work_graph_module.WorkState.RUNNING.value,
            attempt,
            owner,
        )
        if row is None or tuple(row) != expected:
            connection.rollback()
            raise work_graph_module.WorkGraphError(
                f"Stale worker claim rejected for {node_id}: expected "
                f"attempt={attempt}, owner={owner}."
            )

        if shared_index is not None:
            # Generator receipts are nested (page -> child receipts -> operations).
            # Use the canonical recursive extractor rather than looking only at the
            # top-level touched_paths/written_files fields. The index publication stays
            # inside the exact attempt/owner SQLite fence.
            from .scheduler_parallel_safety_contract import _receipt_touched_paths

            touched = _receipt_touched_paths(receipt)
            if touched:
                try:
                    with _INDEX_COMMIT_LOCK:
                        shared_index.update_files(touched)
                        shared_index.write_manifest()
                except Exception as exc:
                    connection.rollback()
                    raise orchestrator_module.CompleteProductionError(
                        f"Shared ProjectIndex commit failed for {node_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

        cursor = connection.execute(
            """
            UPDATE tasks
            SET state = ?, output_hash = ?, receipt_json = ?, receipt_hash = ?,
                lease_owner = NULL, lease_until = NULL, error = NULL,
                updated_at = ?
            WHERE node_id = ? AND state = ? AND attempt = ? AND lease_owner = ?
            """,
            (
                work_graph_module.WorkState.SUCCEEDED.value,
                digest,
                rendered,
                digest,
                time.time(),
                node_id,
                work_graph_module.WorkState.RUNNING.value,
                attempt,
                owner,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise work_graph_module.WorkGraphError(
                f"Work claim changed while succeeding: {node_id}."
            )
        connection.commit()


def install(*, work_graph_module: Any, orchestrator_module: Any) -> None:
    """Fence each generation result to the exact claim that launched it.

    A task can be reclaimed after a lease expires. Looking only at ``state=RUNNING``
    lets a slow result from attempt N publish into attempt N+1. Capture the task's
    ``attempt`` and ``lease_owner`` before invoking the action and require both values
    at publication time. The shared ProjectIndex is committed under the same SQLite
    write transaction so a stale attempt cannot publish index state before its durable
    completion is rejected.
    """

    cls = orchestrator_module.CompleteProductionOrchestrator
    current = cls._run_work_node
    if getattr(current, "_mmm_claim_fenced", False):
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
        if current_task["state"] in {
            work_graph_module.WorkState.FAILED.value,
            work_graph_module.WorkState.INPUT_REQUIRED.value,
            work_graph_module.WorkState.CANCELLED.value,
        }:
            ledger.retry(node.node_id)
            current_task = ledger.task(node.node_id)

        ledger.raise_if_cancelled()
        if current_task["state"] != work_graph_module.WorkState.RUNNING.value:
            ledger.begin(node.node_id, worker_id="complete-orchestrator")

        claim_attempt, claim_owner = _snapshot_claim(
            work_graph_module,
            ledger,
            node.node_id,
        )

        try:
            project_root = (
                getattr(shared_index, "root", None)
                if node.resource_class == "commit" and shared_index is not None
                else None
            )

            def execute_and_commit() -> dict[str, Any]:
                receipt = action()
                if not isinstance(receipt, dict):
                    raise orchestrator_module.CompleteProductionError(
                        f"Work node {node.node_id} returned a non-object receipt."
                    )
                ledger.raise_if_cancelled()
                _commit_success(
                    work_graph_module,
                    orchestrator_module,
                    ledger,
                    node.node_id,
                    receipt,
                    attempt=claim_attempt,
                    owner=claim_owner,
                    shared_index=shared_index,
                )
                return receipt

            if project_root is not None:
                with project_write_lock(project_root):
                    return execute_and_commit()
            return execute_and_commit()
        except BaseException as exc:
            _fenced_fail(
                work_graph_module,
                ledger,
                node.node_id,
                attempt=claim_attempt,
                owner=claim_owner,
                error=exc,
            )
            raise

    run_work_node._mmm_claim_fenced = True  # type: ignore[attr-defined]
    cls._run_work_node = staticmethod(run_work_node)


__all__ = ["install"]
