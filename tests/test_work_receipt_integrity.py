from __future__ import annotations

import sqlite3
from pathlib import Path

from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphPlan,
    WorkNode,
    WorkState,
)


def _ledger(tmp_path: Path) -> DurableWorkLedger:
    ledger = DurableWorkLedger(
        tmp_path / "work.sqlite",
        proposal_hash="sha256:proposal",
    )
    ledger.sync_plan(
        WorkGraphPlan(
            schema_version="mmm/production-work-graph-v1",
            proposal_hash="sha256:proposal",
            graph_hash="sha256:graph",
            module_count=2,
            nodes=(
                WorkNode("a", "generate", "sha256:a", (), {"kind": "a"}),
                WorkNode("b", "validate", "sha256:b", ("a",), {"kind": "b"}),
            ),
        )
    )
    return ledger


def test_receipt_hash_is_independent_from_artifact_output_hash(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    ledger.succeed("a", {"files": ["a.java"]}, output_hash="sha256:artifact")

    with sqlite3.connect(ledger.path) as connection:
        output_hash, receipt_hash = connection.execute(
            "SELECT output_hash, receipt_hash FROM tasks WHERE node_id = 'a'"
        ).fetchone()

    assert output_hash == "sha256:artifact"
    assert receipt_hash.startswith("sha256:")
    assert receipt_hash != output_hash
    assert ledger.cached_receipt("a") == {"files": ["a.java"]}


def test_corrupt_task_receipt_is_invalidated_with_descendants_on_resume(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    ledger.succeed("a", {"value": "a"})
    ledger.begin("b")
    ledger.succeed("b", {"value": "b"})

    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE tasks SET receipt_json = ? WHERE node_id = 'a'",
            ('{"value":"tampered"}',),
        )
        connection.commit()

    reopened = DurableWorkLedger(
        ledger.path,
        proposal_hash="sha256:proposal",
    )

    assert reopened.task("a")["state"] == WorkState.PENDING.value
    assert reopened.task("b")["state"] == WorkState.PENDING.value
    assert reopened.cached_receipt("a") is None
    assert reopened.cached_receipt("b") is None


def test_legacy_receipt_is_promoted_only_when_old_hash_proves_same_payload(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    ledger.succeed("a", {"value": "legacy"})
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("UPDATE tasks SET receipt_hash = NULL WHERE node_id = 'a'")
        connection.commit()

    reopened = DurableWorkLedger(
        ledger.path,
        proposal_hash="sha256:proposal",
    )

    assert reopened.cached_receipt("a") == {"value": "legacy"}
    with sqlite3.connect(ledger.path) as connection:
        receipt_hash = connection.execute(
            "SELECT receipt_hash FROM tasks WHERE node_id = 'a'"
        ).fetchone()[0]
    assert receipt_hash.startswith("sha256:")


def test_unverifiable_legacy_custom_output_hash_cannot_survive_resume(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    ledger.succeed("a", {"value": "legacy"}, output_hash="sha256:artifact")
    with sqlite3.connect(ledger.path) as connection:
        connection.execute("UPDATE tasks SET receipt_hash = NULL WHERE node_id = 'a'")
        connection.commit()

    reopened = DurableWorkLedger(
        ledger.path,
        proposal_hash="sha256:proposal",
    )

    assert reopened.task("a")["state"] == WorkState.PENDING.value
    assert reopened.cached_receipt("a") is None


def test_corrupt_checkpoint_receipt_is_failed_and_cannot_be_reused(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin_checkpoint("checkpoint", stage="plan", input_hash="sha256:input")
    ledger.succeed_checkpoint(
        "checkpoint",
        input_hash="sha256:input",
        receipt={"plan": "trusted"},
    )
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE checkpoints SET receipt_json = ? WHERE checkpoint_id = 'checkpoint'",
            ('{"plan":"tampered"}',),
        )
        connection.commit()

    reopened = DurableWorkLedger(
        ledger.path,
        proposal_hash="sha256:proposal",
    )

    assert reopened.cached_checkpoint(
        "checkpoint", input_hash="sha256:input"
    ) is None
    with sqlite3.connect(ledger.path) as connection:
        state, receipt_json, receipt_hash = connection.execute(
            "SELECT state, receipt_json, receipt_hash FROM checkpoints "
            "WHERE checkpoint_id = 'checkpoint'"
        ).fetchone()
    assert state == WorkState.FAILED.value
    assert receipt_json is None
    assert receipt_hash is None
