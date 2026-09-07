from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

from minecraft_mod_ai import execution_feedback_replan_contract as feedback
from minecraft_mod_ai.remote_trajectory_store import _stamp_remote_record
from minecraft_mod_ai.trajectory_memory import build_work_trajectory
from minecraft_mod_ai.trajectory_record_integrity import (
    record_remote_eligible,
    validate_trajectory_record,
)
from minecraft_mod_ai.trajectory_verification import classify_verification
from minecraft_mod_ai.work_graph import (
    DurableWorkLedger,
    WorkGraphPlan,
    WorkNode,
    WorkState,
)


def _task() -> dict[str, object]:
    return {
        "node_id": "repair-demo",
        "stage": "repair",
        "payload": {"kind": "custom_java", "members": [{"module_id": "demo"}]},
    }


def _strong_receipt() -> dict[str, object]:
    return {
        "build": {
            "commands": [
                {"name": "clean_build", "exit_code": 0, "timed_out": False},
                {"name": "gametest", "exit_code": 0, "timed_out": False},
            ]
        }
    }


def test_producer_does_not_promote_contradictory_build_evidence() -> None:
    receipt = _strong_receipt()
    receipt["prior_attempt"] = {
        "commands": [
            {"name": "gradle_build", "exit_code": 1, "timed_out": False}
        ]
    }

    verification = classify_verification(
        task_class="repair",
        outcome="SUCCESS",
        receipt=receipt,
    )

    assert verification["checks"]["build"] is False
    assert verification["level_index"] < 3
    assert verification["strong_skill_eligible"] is False
    assert verification["remote_eligible"] is False


def test_rehashed_remote_record_cannot_hide_conflicting_verifier_evidence() -> None:
    row = build_work_trajectory(
        _task(),
        outcome="SUCCESS",
        receipt=_strong_receipt(),
    )
    assert validate_trajectory_record(row)

    tampered = copy.deepcopy(row)
    verification = tampered["verification"]
    assert isinstance(verification, dict)
    chain = verification["verifier_chain"]
    assert isinstance(chain, list)
    chain.append(
        {
            "kind": "build",
            "status": "FAIL",
            "source": "adversarial-rehashed-record",
        }
    )

    remote = _stamp_remote_record(tampered)
    assert not validate_trajectory_record(remote)
    assert not record_remote_eligible(remote)


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
            module_count=1,
            nodes=(
                WorkNode("a", "generate:test", "sha256:a", (), {"kind": "a"}),
            ),
        )
    )
    return ledger


def test_portable_export_keeps_receipt_hash_distinct_from_artifact_hash(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    receipt = {"files": ["a.java"], "status": "PASS"}
    ledger.succeed("a", receipt, output_hash="sha256:artifact")

    exported = ledger.export_receipts(tmp_path / "receipts.jsonl")
    rows = [json.loads(line) for line in exported.read_text(encoding="utf-8").splitlines()]
    task = next(row for row in rows if row.get("record_type") == "task")

    expected_receipt_json = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_receipt_hash = "sha256:" + hashlib.sha256(
        expected_receipt_json.encode("utf-8")
    ).hexdigest()

    assert task["receipt"] == receipt
    assert task["output_hash"] == "sha256:artifact"
    assert task["receipt_hash"] == expected_receipt_hash
    assert task["receipt_hash"] != task["output_hash"]


def test_task_page_invalidates_receipt_tampered_after_runtime_audit(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    ledger.begin("a")
    ledger.succeed(
        "a",
        {
            "semantic_observations": [
                {
                    "task_ids": ["trusted-owner"],
                    "touched_paths": ["src/main/java/Trusted.java"],
                }
            ]
        },
    )

    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE tasks SET receipt_json = ? WHERE node_id = 'a'",
            (
                json.dumps(
                    {
                        "semantic_observations": [
                            {
                                "task_ids": ["forged-owner"],
                                "touched_paths": ["src/main/java/Forged.java"],
                            }
                        ]
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
        connection.commit()

    page = ledger.tasks(limit=10)
    task = next(row for row in page["tasks"] if row["node_id"] == "a")

    assert task["state"] == WorkState.PENDING.value
    assert task["receipt"] is None
    assert ledger.cached_receipt("a") is None


def _validation_receipt() -> dict[str, object]:
    return {
        "status": "FAIL",
        "diagnostics": [
            {
                "path": "src/main/java/Demo.java",
                "message": "cannot resolve symbol",
                "code": "E100",
                "severity": 1,
            }
        ],
    }


def _validation_checkpoint(ledger: DurableWorkLedger) -> None:
    ledger.begin_checkpoint(
        "validate-jdt",
        stage="validate:jdt",
        input_hash="sha256:validation-input",
    )
    ledger.succeed_checkpoint(
        "validate-jdt",
        input_hash="sha256:validation-input",
        receipt=_validation_receipt(),
    )


def test_replan_reader_rejects_checkpoint_tampered_after_runtime_audit(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    _validation_checkpoint(ledger)
    with sqlite3.connect(ledger.path) as connection:
        connection.execute(
            "UPDATE checkpoints SET receipt_json = ? WHERE checkpoint_id = 'validate-jdt'",
            ('{"status":"FAIL","diagnostics":[{"message":"forged"}]}',),
        )
        connection.commit()

    try:
        raise RuntimeError("JDT reported errors")
    except RuntimeError:
        current = feedback._latest_failed_feedback(ledger)

    assert current is None
    assert ledger.cached_checkpoint(
        "validate-jdt",
        input_hash="sha256:validation-input",
    ) is None
    with sqlite3.connect(ledger.path) as connection:
        state, receipt_json, receipt_hash = connection.execute(
            "SELECT state, receipt_json, receipt_hash FROM checkpoints "
            "WHERE checkpoint_id = 'validate-jdt'"
        ).fetchone()
    assert state == WorkState.FAILED.value
    assert receipt_json is None
    assert receipt_hash is None


def test_replan_reader_accepts_verified_current_exception_checkpoint(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path)
    _validation_checkpoint(ledger)

    try:
        raise RuntimeError("JDT reported errors")
    except RuntimeError:
        current = feedback._latest_failed_feedback(ledger)

    assert current is not None
    assert current["checkpoint_id"] == "validate-jdt"
    assert current["failure_scope"] == "current_exception"
    assert current["diagnostics"]
