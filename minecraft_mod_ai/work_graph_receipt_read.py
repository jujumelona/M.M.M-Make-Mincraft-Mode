from __future__ import annotations

"""Read-boundary integrity for durable work receipts without runtime rebinding."""

import sqlite3
from collections.abc import Sequence
from typing import Any


_TASK_SELECT = """
    SELECT node_id, stage, input_hash, payload_json, state,
           attempt, lease_owner, lease_until, output_hash,
           receipt_json, receipt_hash, error, updated_at
    FROM tasks WHERE node_id = ?
"""


def _select_task(
    connection: sqlite3.Connection,
    node_id: str,
    *,
    error_type: type[Exception],
) -> Sequence[Any]:
    row = connection.execute(_TASK_SELECT, (node_id,)).fetchone()
    if row is None:
        raise error_type(f"Unknown work node: {node_id}")
    return row


def _receipt_status(ledger: Any, row: Sequence[Any]) -> str:
    if str(row[4]) != "succeeded":
        return "not_succeeded"
    digest = ledger._verified_receipt_digest(row[9])
    if digest is None:
        return "invalid"
    receipt_hash = str(row[10] or "")
    if receipt_hash:
        return "valid" if receipt_hash == digest else "invalid"
    return "legacy" if str(row[8] or "") == digest else "invalid"


def verified_task_row(
    ledger: Any,
    connection: sqlite3.Connection,
    node_id: str,
    *,
    error_type: type[Exception],
) -> Sequence[Any]:
    row = _select_task(connection, node_id, error_type=error_type)
    status = _receipt_status(ledger, row)
    if status == "invalid":
        ledger._invalidate_many(connection, [node_id])
        connection.commit()
        return _select_task(connection, node_id, error_type=error_type)
    if status == "legacy":
        digest = ledger._verified_receipt_digest(row[9])
        connection.execute(
            "UPDATE tasks SET receipt_hash = ? WHERE node_id = ?",
            (digest, node_id),
        )
        connection.commit()
        return _select_task(connection, node_id, error_type=error_type)
    return row


def verified_task_page(
    ledger: Any,
    connection: sqlite3.Connection,
    rows: Sequence[Sequence[Any]],
    *,
    reread: Any,
) -> Sequence[Sequence[Any]]:
    """Verify a whole selected page in one pass and reread after invalidation.

    A task list is a production ownership/repair input, not merely display
    metadata. Never surface stale semantic observations from a modified receipt.
    """
    corrupted: list[str] = []
    legacy: list[tuple[str, str]] = []
    for row in rows:
        status = _receipt_status(ledger, row)
        if status == "invalid":
            corrupted.append(str(row[0]))
        elif status == "legacy":
            digest = ledger._verified_receipt_digest(row[9])
            if digest is not None:
                legacy.append((digest, str(row[0])))
    if corrupted:
        ledger._invalidate_many(connection, corrupted)
    for digest, node_id in legacy:
        if node_id not in corrupted:
            connection.execute(
                "UPDATE tasks SET receipt_hash = ? WHERE node_id = ? "
                "AND state = 'succeeded'",
                (digest, node_id),
            )
    if corrupted or legacy:
        connection.commit()
        return reread()
    return rows


def verified_checkpoint_row(
    ledger: Any,
    connection: sqlite3.Connection,
    checkpoint_id: str,
) -> Sequence[Any] | None:
    row = connection.execute(
        "SELECT state, input_hash, receipt_json, receipt_hash, output_hash "
        "FROM checkpoints WHERE checkpoint_id = ?",
        (checkpoint_id,),
    ).fetchone()
    if row is None or str(row[0]) != "succeeded":
        return row
    digest = ledger._verified_receipt_digest(row[2])
    valid = digest is not None and (
        str(row[3]) == digest if row[3] else str(row[4] or "") == digest
    )
    if not valid:
        connection.execute(
            "UPDATE checkpoints SET state = 'failed', receipt_json = NULL, "
            "receipt_hash = NULL, output_hash = NULL, error = ?, "
            "updated_at = strftime('%s','now') WHERE checkpoint_id = ? "
            "AND state = 'succeeded'",
            ("Stored checkpoint receipt failed integrity verification.", checkpoint_id),
        )
        connection.commit()
        return None
    if not row[3]:
        connection.execute(
            "UPDATE checkpoints SET receipt_hash = ? WHERE checkpoint_id = ? "
            "AND state = 'succeeded'",
            (digest, checkpoint_id),
        )
        connection.commit()
    return row


__all__ = ["verified_task_row", "verified_task_page", "verified_checkpoint_row"]
