from __future__ import annotations

"""Claim-fenced publication helpers for the canonical orchestrator work-node owner."""

import hashlib
import threading
import time
from typing import Any

_INDEX_COMMIT_LOCK = threading.RLock()


def _snapshot_claim(ledger: Any, node_id: str) -> tuple[int, str]:
    from .work_graph import WorkGraphError, WorkState

    current = ledger.task(node_id)
    if current["state"] != WorkState.RUNNING.value:
        raise WorkGraphError(
            f"Work node {node_id} is not running while capturing its claim: "
            f"{current['state']}"
        )
    attempt = current.get("attempt")
    owner = current.get("lease_owner")
    if type(attempt) is not int or attempt < 1:
        raise WorkGraphError(f"Work node {node_id} has an invalid running attempt.")
    if not isinstance(owner, str) or not owner:
        raise WorkGraphError(f"Work node {node_id} has no running owner.")
    return attempt, owner


def _fenced_fail(
    ledger: Any,
    node_id: str,
    *,
    attempt: int,
    owner: str,
    error: BaseException,
) -> None:
    from .work_graph import WorkState

    with ledger._connect() as connection:
        cursor = connection.execute(
            """
            UPDATE tasks
            SET state = ?, error = ?, lease_owner = NULL,
                lease_until = NULL, updated_at = ?
            WHERE node_id = ? AND state = ? AND attempt = ? AND lease_owner = ?
            """,
            (
                WorkState.FAILED.value,
                f"{type(error).__name__}: {error}"[:16_384],
                time.time(),
                node_id,
                WorkState.RUNNING.value,
                attempt,
                owner,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return
        connection.commit()


def _commit_success(
    ledger: Any,
    node_id: str,
    receipt: dict[str, Any],
    *,
    attempt: int,
    owner: str,
    shared_index: Any | None,
    index_error_type: type[Exception] = RuntimeError,
) -> None:
    from .scheduler_parallel_safety_contract import _receipt_touched_paths
    from .work_graph import WorkGraphError, WorkState, canonical_json

    rendered = canonical_json(receipt)
    digest = "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    with ledger._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT state, attempt, lease_owner FROM tasks WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        expected = (WorkState.RUNNING.value, attempt, owner)
        if row is None or tuple(row) != expected:
            connection.rollback()
            raise WorkGraphError(
                f"Stale worker claim rejected for {node_id}: expected "
                f"attempt={attempt}, owner={owner}."
            )

        if shared_index is not None:
            touched = _receipt_touched_paths(receipt)
            if touched:
                try:
                    with _INDEX_COMMIT_LOCK:
                        shared_index.update_files(touched)
                        shared_index.write_manifest()
                except Exception as exc:
                    connection.rollback()
                    raise index_error_type(
                        f"Shared ProjectIndex commit failed for {node_id}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc

        cursor = connection.execute(
            """
            UPDATE tasks
            SET state = ?, output_hash = ?, receipt_json = ?,
                lease_owner = NULL, lease_until = NULL, error = NULL,
                updated_at = ?
            WHERE node_id = ? AND state = ? AND attempt = ? AND lease_owner = ?
            """,
            (
                WorkState.SUCCEEDED.value,
                digest,
                rendered,
                time.time(),
                node_id,
                WorkState.RUNNING.value,
                attempt,
                owner,
            ),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            raise WorkGraphError(f"Work claim changed while succeeding: {node_id}.")
        connection.commit()


__all__ = ["_commit_success", "_fenced_fail", "_snapshot_claim"]
