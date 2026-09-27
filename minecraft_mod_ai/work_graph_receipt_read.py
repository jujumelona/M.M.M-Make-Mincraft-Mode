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


__all__ = ["verified_task_row"]
