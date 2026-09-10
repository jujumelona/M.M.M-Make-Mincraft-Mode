from __future__ import annotations

import pytest

from minecraft_mod_ai.reuse_template_pipeline import (
    REUSE_SEQUENCE,
    evaluate_feature_reuse,
    evaluate_reuse_pipeline,
    execute_reuse_template,
    validate_reuse_template_sequence,
)


def test_reuse_template_sequence_is_canonical():
    validate_reuse_template_sequence()
    assert len(REUSE_SEQUENCE) == 15
    assert REUSE_SEQUENCE[0] == "reuse/query_build"
    assert REUSE_SEQUENCE[-1] == "reuse/integration"


def test_evaluate_feature_reuse_generates_all_receipts():
    feature = {
        "feature_id": "energy_storage",
        "feature_description": "Store energy units in block entity",
    }
    saved_progress = {}

    def checkpoint(binding, receipt):
        saved_progress[binding] = receipt

    result = evaluate_feature_reuse(feature, checkpoint=checkpoint)
    assert result["feature_id"] == "energy_storage"
    assert len(result["receipts"]) == 15
    assert len(saved_progress) == 15

    receipt_ids = [r["template_id"] for r in result["receipts"]]
    assert tuple(receipt_ids) == REUSE_SEQUENCE
    assert all(r["status"] == "PASS" for r in result["receipts"])
    assert all(r["proof"]["passed"] is True for r in result["receipts"])

    # Replay with saved progress
    replayed = evaluate_feature_reuse(feature, progress=saved_progress)
    assert replayed["receipts"] == result["receipts"]


def test_evaluate_reuse_pipeline_runs_for_each_atomic_feature():
    features = [
        {"feature_id": "feat_a", "feature_description": "Description A"},
        {"feature_id": "feat_b", "feature_description": "Description B"},
    ]
    results = evaluate_reuse_pipeline(features)
    assert len(results) == 2
    assert [r["feature_id"] for r in results] == ["feat_a", "feat_b"]
    assert all(len(r["receipts"]) == 15 for r in results)


def test_reuse_proof_blocking():
    # 1. Unresolved license terms block reuse/license_check
    r1 = execute_reuse_template(
        "reuse/license_check",
        context={"unresolved_terms": ["Proprietary no-redistribution clause"]},
    )
    assert r1["status"] == "BLOCKED"
    assert r1["proof"]["passed"] is False

    # 2. Failed compatibility checks block reuse/compatibility_check
    r2 = execute_reuse_template(
        "reuse/compatibility_check",
        context={"failed_checks": ["Target Minecraft version mismatch (1.16 vs 1.20)"]},
    )
    assert r2["status"] == "BLOCKED"
    assert r2["proof"]["passed"] is False
    assert r2["output"]["compatible"] is False

    # 3. Blocking reasons block reuse/direct_reuse
    r3 = execute_reuse_template(
        "reuse/direct_reuse",
        context={"blocking_reasons": ["GPL license incompatible with MIT target"]},
    )
    assert r3["status"] == "BLOCKED"
    assert r3["proof"]["passed"] is False
    assert r3["output"]["reusable_directly"] is False
