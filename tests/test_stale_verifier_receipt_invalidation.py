from __future__ import annotations

import json

import minecraft_mod_ai.work_graph as work_graph
from minecraft_mod_ai.work_graph_receipt_integrity_contract import (
    _receipt_hash,
    install as install_receipt_integrity,
)


def _ledger_with_verified_source_receipt(tmp_path):
    install_receipt_integrity(work_graph)
    plan = work_graph.WorkGraphPlan(
        schema_version="mmm/production-work-graph-v1",
        proposal_hash="sha256:proposal",
        graph_hash="sha256:graph",
        module_count=0,
        nodes=(
            work_graph.WorkNode(
                node_id="validate-source",
                stage="validate:source",
                input_hash="sha256:source-input",
                dependencies=(),
                payload={"kind": "validation", "validator_config": {"strict": True}},
            ),
        ),
    )
    ledger = work_graph.DurableWorkLedger(
        tmp_path / "work.sqlite3",
        proposal_hash=plan.proposal_hash,
    )
    ledger.sync_plan(plan)
    ledger.begin("validate-source")
    ledger.succeed(
        "validate-source",
        {
            "status": "PASS",
            "checks_run": 3,
            "project_manifest": "sha256:project-tree",
        },
    )
    return ledger


def _rewrite_receipt_with_valid_integrity(ledger, mutate):
    with ledger._connect() as connection:
        row = connection.execute(
            "SELECT receipt_json FROM tasks WHERE node_id = ?",
            ("validate-source",),
        ).fetchone()
        receipt = json.loads(row[0])
        mutate(receipt)
        rendered = work_graph.canonical_json(receipt)
        connection.execute(
            "UPDATE tasks SET receipt_json = ?, receipt_hash = ? WHERE node_id = ?",
            (rendered, _receipt_hash(rendered), "validate-source"),
        )
        connection.commit()


def _raw_state(ledger) -> str:
    with ledger._connect() as connection:
        row = connection.execute(
            "SELECT state FROM tasks WHERE node_id = ?",
            ("validate-source",),
        ).fetchone()
    return str(row[0])


def test_cached_receipt_rejects_cryptographically_valid_stale_verifier_version(tmp_path):
    ledger = _ledger_with_verified_source_receipt(tmp_path)

    def stale_verifier(receipt):
        receipt["_mmm_completion_evidence"]["verifier_version_hash"] = "sha256:stale-verifier"

    _rewrite_receipt_with_valid_integrity(ledger, stale_verifier)

    assert ledger.cached_receipt(
        "validate-source",
        input_hash="sha256:source-input",
    ) is None
    assert _raw_state(ledger) == work_graph.WorkState.PENDING.value


def test_resume_invalidates_stale_verifier_config_before_reuse(tmp_path):
    ledger = _ledger_with_verified_source_receipt(tmp_path)

    def stale_config(receipt):
        receipt["_mmm_completion_evidence"]["verifier_config_hash"] = "sha256:old-config"

    _rewrite_receipt_with_valid_integrity(ledger, stale_config)

    ledger.resume_run()

    assert _raw_state(ledger) == work_graph.WorkState.PENDING.value
    assert ledger.cached_receipt("validate-source") is None


def test_legacy_verified_receipt_without_reuse_fingerprints_is_invalidated(tmp_path):
    ledger = _ledger_with_verified_source_receipt(tmp_path)

    def legacy_evidence(receipt):
        evidence = receipt["_mmm_completion_evidence"]
        receipt["_mmm_completion_evidence"] = {
            "schema_version": "mmm/work-completion-evidence-v1",
            "node_id": evidence["node_id"],
            "stage": evidence["stage"],
            "input_hash": evidence["input_hash"],
            "completion_scope": evidence["completion_scope"],
            "verifier": evidence["verifier"],
        }

    _rewrite_receipt_with_valid_integrity(ledger, legacy_evidence)

    assert ledger.cached_receipt("validate-source") is None
    assert _raw_state(ledger) == work_graph.WorkState.PENDING.value
