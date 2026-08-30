from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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
                WorkNode("a", "generate", "sha256:a", (), {"kind": "a"}),
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
