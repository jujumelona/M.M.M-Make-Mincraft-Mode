from __future__ import annotations

"""Bind durable work state to independently verified receipt identity.

``output_hash`` may identify a produced artifact, so it must not double as the
integrity proof for ``receipt_json``. This contract adds a separate receipt hash,
migrates only legacy rows whose old output hash actually proves the receipt body,
and invalidates unverifiable successful state at resume/read boundaries.
"""

import hashlib
import hmac
import json
import sqlite3
import time
from functools import wraps
from pathlib import Path
from typing import Any

_INSTALLED = False
_INTEGRITY_ERROR = "receipt integrity verification failed"


def _receipt_hash(receipt_json: str) -> str:
    return "sha256:" + hashlib.sha256(receipt_json.encode("utf-8")).hexdigest()


def _verified_receipt(
    receipt_json: str | None,
    receipt_hash: str | None,
    *,
    label: str,
    error_type: type[Exception],
) -> dict[str, Any]:
    if not receipt_json or not receipt_hash:
        raise error_type(f"{label} has no authoritative receipt integrity proof.")
    expected = _receipt_hash(receipt_json)
    if not hmac.compare_digest(str(receipt_hash), expected):
        raise error_type(f"{label} receipt hash does not match its persisted payload.")
    try:
        value = json.loads(receipt_json)
    except json.JSONDecodeError as exc:
        raise error_type(f"{label} receipt is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise error_type(f"{label} receipt must be a JSON object.")
    return value


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _ensure_schema(connection: sqlite3.Connection) -> None:
    for table in ("tasks", "checkpoints"):
        if "receipt_hash" not in _column_names(connection, table):
            connection.execute(f"ALTER TABLE {table} ADD COLUMN receipt_hash TEXT")
    connection.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS tasks_clear_receipt_hash
        AFTER UPDATE OF receipt_json ON tasks
        WHEN NEW.receipt_json IS NULL AND NEW.receipt_hash IS NOT NULL
        BEGIN
            UPDATE tasks SET receipt_hash = NULL WHERE node_id = NEW.node_id;
        END;
        CREATE TRIGGER IF NOT EXISTS checkpoints_clear_receipt_hash
        AFTER UPDATE OF receipt_json ON checkpoints
        WHEN NEW.receipt_json IS NULL AND NEW.receipt_hash IS NOT NULL
        BEGIN
            UPDATE checkpoints
            SET receipt_hash = NULL
            WHERE checkpoint_id = NEW.checkpoint_id;
        END;
        """
    )


def _audit_legacy_receipts(ledger: Any, module: Any, connection: sqlite3.Connection) -> None:
    succeeded = module.WorkState.SUCCEEDED.value
    task_invalid: list[str] = []
    for node_id, output_hash, receipt_json, receipt_hash in connection.execute(
        """
        SELECT node_id, output_hash, receipt_json, receipt_hash
        FROM tasks WHERE state = ?
        """,
        (succeeded,),
    ):
        if not receipt_json:
            task_invalid.append(str(node_id))
            continue
        expected = _receipt_hash(str(receipt_json))
        if receipt_hash and hmac.compare_digest(str(receipt_hash), expected):
            continue
        if not receipt_hash and output_hash and hmac.compare_digest(str(output_hash), expected):
            connection.execute(
                "UPDATE tasks SET receipt_hash = ? WHERE node_id = ?",
                (expected, node_id),
            )
            continue
        task_invalid.append(str(node_id))

    if task_invalid:
        ledger._invalidate_many(connection, task_invalid)

    failed = module.WorkState.FAILED.value
    now = time.time()
    for checkpoint_id, output_hash, receipt_json, receipt_hash in connection.execute(
        """
        SELECT checkpoint_id, output_hash, receipt_json, receipt_hash
        FROM checkpoints WHERE state = ?
        """,
        (succeeded,),
    ).fetchall():
        if receipt_json:
            expected = _receipt_hash(str(receipt_json))
            if receipt_hash and hmac.compare_digest(str(receipt_hash), expected):
                continue
            if not receipt_hash and output_hash and hmac.compare_digest(str(output_hash), expected):
                connection.execute(
                    "UPDATE checkpoints SET receipt_hash = ? WHERE checkpoint_id = ?",
                    (expected, checkpoint_id),
                )
                continue
        connection.execute(
            """
            UPDATE checkpoints
            SET state = ?, output_hash = NULL, receipt_json = NULL,
                receipt_hash = NULL, error = ?, updated_at = ?
            WHERE checkpoint_id = ?
            """,
            (failed, _INTEGRITY_ERROR, now, checkpoint_id),
        )


def _audit_integrity(ledger: Any, module: Any) -> None:
    with ledger._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _ensure_schema(connection)
        _audit_legacy_receipts(ledger, module, connection)
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
            ("receipt_integrity", "sha256-v1"),
        )
        connection.commit()


def install(work_graph_module: Any) -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    ledger_cls = work_graph_module.DurableWorkLedger
    error_type = work_graph_module.WorkGraphError
    canonical_json = work_graph_module.canonical_json
    original_initialize = ledger_cls._initialize
    original_resume_run = ledger_cls.resume_run
    original_export_receipts = ledger_cls.export_receipts
    original_task = ledger_cls.task
    original_tasks = ledger_cls.tasks

    @wraps(original_initialize)
    def initialize(self: Any) -> None:
        original_initialize(self)
        _audit_integrity(self, work_graph_module)

    @wraps(ledger_cls.succeed)
    def succeed(
        self: Any,
        node_id: str,
        receipt: dict[str, Any],
        *,
        output_hash: str = "",
    ) -> dict[str, Any]:
        receipt_json = canonical_json(receipt)
        receipt_hash = _receipt_hash(receipt_json)
        artifact_hash = output_hash or receipt_hash
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state, output_hash, receipt_json, receipt_hash
                FROM tasks WHERE node_id = ?
                """,
                (node_id,),
            ).fetchone()
            if row is None:
                raise error_type(f"Unknown work node: {node_id}")

            state = str(row[0])
            if state == work_graph_module.WorkState.SUCCEEDED.value:
                if (
                    str(row[1] or "") == artifact_hash
                    and str(row[2] or "") == receipt_json
                    and str(row[3] or "") == receipt_hash
                ):
                    connection.commit()
                    return self.task(node_id)
                raise error_type(
                    f"Work node {node_id} already succeeded with a different receipt."
                )

            if state != work_graph_module.WorkState.RUNNING.value:
                raise error_type(
                    f"Work node {node_id} cannot succeed from state {state}."
                )

            cursor = connection.execute(
                """
                UPDATE tasks
                SET state = ?, output_hash = ?, receipt_json = ?, receipt_hash = ?,
                    lease_owner = NULL, lease_until = NULL, error = NULL,
                    updated_at = ?
                WHERE node_id = ? AND state = ?
                """,
                (
                    work_graph_module.WorkState.SUCCEEDED.value,
                    artifact_hash,
                    receipt_json,
                    receipt_hash,
                    time.time(),
                    node_id,
                    work_graph_module.WorkState.RUNNING.value,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                raise error_type(f"Work node changed while succeeding: {node_id}")
            connection.commit()
        return self.task(node_id)

    @wraps(ledger_cls.succeed_checkpoint)
    def succeed_checkpoint(
        self: Any,
        checkpoint_id: str,
        *,
        input_hash: str,
        receipt: dict[str, Any],
    ) -> None:
        rendered = canonical_json(receipt)
        receipt_hash = _receipt_hash(rendered)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE checkpoints
                SET state = ?, receipt_json = ?, output_hash = ?, receipt_hash = ?,
                    error = NULL, updated_at = ?
                WHERE checkpoint_id = ? AND input_hash = ? AND state = ?
                """,
                (
                    work_graph_module.WorkState.SUCCEEDED.value,
                    rendered,
                    receipt_hash,
                    receipt_hash,
                    time.time(),
                    checkpoint_id,
                    input_hash,
                    work_graph_module.WorkState.RUNNING.value,
                ),
            )
            if cursor.rowcount == 0:
                raise error_type(f"Checkpoint changed while running: {checkpoint_id}")
            connection.commit()

    @wraps(ledger_cls.cached_receipt)
    def cached_receipt(
        self: Any,
        node_id: str,
        *,
        input_hash: str | None = None,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state, input_hash, receipt_json, receipt_hash
                FROM tasks WHERE node_id = ?
                """,
                (node_id,),
            ).fetchone()
        if row is None or row[0] != work_graph_module.WorkState.SUCCEEDED.value:
            return None
        if input_hash is not None and row[1] != input_hash:
            return None
        try:
            return _verified_receipt(
                row[2], row[3], label=f"Work node {node_id}", error_type=error_type
            )
        except error_type:
            self.invalidate(node_id)
            return None

    @wraps(ledger_cls.cached_checkpoint)
    def cached_checkpoint(
        self: Any,
        checkpoint_id: str,
        *,
        input_hash: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT state, input_hash, receipt_json, receipt_hash
                FROM checkpoints WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            ).fetchone()
        if (
            row is None
            or row[0] != work_graph_module.WorkState.SUCCEEDED.value
            or row[1] != input_hash
        ):
            return None
        try:
            return _verified_receipt(
                row[2],
                row[3],
                label=f"Checkpoint {checkpoint_id}",
                error_type=error_type,
            )
        except error_type:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    UPDATE checkpoints
                    SET state = ?, output_hash = NULL, receipt_json = NULL,
                        receipt_hash = NULL, error = ?, updated_at = ?
                    WHERE checkpoint_id = ?
                    """,
                    (
                        work_graph_module.WorkState.FAILED.value,
                        _INTEGRITY_ERROR,
                        time.time(),
                        checkpoint_id,
                    ),
                )
                connection.commit()
            return None

    def _verified_task_view(self: Any, row: dict[str, Any]) -> dict[str, Any]:
        state = str(row.get("state") or "")
        if state != work_graph_module.WorkState.SUCCEEDED.value:
            if row.get("receipt") is None:
                return row
            sanitized = dict(row)
            sanitized["receipt"] = None
            return sanitized

        node_id = str(row.get("node_id") or "")
        input_hash = str(row.get("input_hash") or "")
        verified = cached_receipt(self, node_id, input_hash=input_hash)
        if verified is None:
            refreshed = original_task(self, node_id)
            refreshed["receipt"] = None
            return refreshed
        result = dict(row)
        result["receipt"] = verified
        return result

    @wraps(original_task)
    def task(self: Any, node_id: str) -> dict[str, Any]:
        return _verified_task_view(self, original_task(self, node_id))

    @wraps(original_tasks)
    def tasks(
        self: Any,
        *,
        cursor: str = "",
        limit: int = 100,
        state: Any = None,
    ) -> dict[str, Any]:
        page = dict(original_tasks(self, cursor=cursor, limit=limit, state=state))
        page["tasks"] = [
            _verified_task_view(self, dict(row))
            for row in page.get("tasks", ())
            if isinstance(row, dict)
        ]
        return page

    @wraps(original_resume_run)
    def resume_run(self: Any) -> dict[str, Any]:
        result = original_resume_run(self)
        _audit_integrity(self, work_graph_module)
        return self.summary() if result is not None else result

    @wraps(original_export_receipts)
    def export_receipts(self: Any, path: str | Path) -> Path:
        """Export portable receipts with the independent receipt integrity hash."""

        _audit_integrity(self, work_graph_module)
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                canonical_json({"record_type": "summary", "value": self.summary()})
                + "\n"
            )
            with self._connect() as connection:
                for row in connection.execute(
                    """
                    SELECT node_id, stage, input_hash, state, attempt,
                           output_hash, receipt_json, receipt_hash, error, updated_at
                    FROM tasks ORDER BY node_id
                    """
                ):
                    stream.write(
                        canonical_json(
                            {
                                "record_type": "task",
                                "node_id": row[0],
                                "stage": row[1],
                                "input_hash": row[2],
                                "state": row[3],
                                "attempt": row[4],
                                "output_hash": row[5],
                                "receipt": json.loads(row[6]) if row[6] else None,
                                "receipt_hash": row[7],
                                "error": row[8],
                                "updated_at": row[9],
                            }
                        )
                        + "\n"
                    )
                for row in connection.execute(
                    """
                    SELECT checkpoint_id, stage, input_hash, state, attempt,
                           output_hash, receipt_json, receipt_hash, error, updated_at
                    FROM checkpoints ORDER BY checkpoint_id
                    """
                ):
                    stream.write(
                        canonical_json(
                            {
                                "record_type": "checkpoint",
                                "checkpoint_id": row[0],
                                "stage": row[1],
                                "input_hash": row[2],
                                "state": row[3],
                                "attempt": row[4],
                                "output_hash": row[5],
                                "receipt": json.loads(row[6]) if row[6] else None,
                                "receipt_hash": row[7],
                                "error": row[8],
                                "updated_at": row[9],
                            }
                        )
                        + "\n"
                    )
        temporary.replace(target)
        return target

    ledger_cls._initialize = initialize
    ledger_cls.succeed = succeed
    ledger_cls.succeed_checkpoint = succeed_checkpoint
    ledger_cls.cached_receipt = cached_receipt
    ledger_cls.cached_checkpoint = cached_checkpoint
    ledger_cls.task = task
    ledger_cls.tasks = tasks
    ledger_cls.resume_run = resume_run
    ledger_cls.export_receipts = export_receipts
    _INSTALLED = True


__all__ = ["install"]
