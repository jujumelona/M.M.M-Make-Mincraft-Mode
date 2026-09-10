from __future__ import annotations

import pytest

from minecraft_mod_ai.reuse_template_pipeline import (
    REUSE_SEQUENCE,
    evaluate_feature_reuse,
    evaluate_reuse_pipeline,
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
