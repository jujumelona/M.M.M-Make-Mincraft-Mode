from __future__ import annotations

import pytest

from minecraft_mod_ai.research_template_pipeline import (
    RESEARCH_SEQUENCE,
    execute_research_template,
    run_research_pipeline,
    validate_research_template_sequence,
)


def test_research_template_sequence_is_canonical():
    validate_research_template_sequence()
    assert len(RESEARCH_SEQUENCE) == 9
    assert RESEARCH_SEQUENCE[0] == "research/reference_identity"
    assert RESEARCH_SEQUENCE[-1] == "research/evidence_check"


def test_run_research_pipeline_generates_all_receipts():
    state = {
        "references": [{"name": "Sample Reference", "what_must_be_learned": "mechanics"}],
    }
    evidence = [
        {"kind": "visual", "source": "screenshot.png", "detail": "blue texture"},
        {"kind": "audio", "source": "sound.ogg", "detail": "wind loop"},
    ]
    saved_progress = {}

    def checkpoint(binding, receipt):
        saved_progress[binding] = receipt

    result = run_research_pipeline(state, evidence_items=evidence, checkpoint=checkpoint)
    assert len(result["receipts"]) == 9
    assert len(saved_progress) == 9

    receipt_ids = [r["template_id"] for r in result["receipts"]]
    assert tuple(receipt_ids) == RESEARCH_SEQUENCE
    assert all(r["status"] == "PASS" for r in result["receipts"])
    assert all(r["proof"]["passed"] is True for r in result["receipts"])

    # Replay with saved progress
    replayed = run_research_pipeline(state, evidence_items=evidence, progress=saved_progress)
    assert replayed["receipts"] == result["receipts"]


def test_research_proof_blocking():
    # 1. Unresolved identities block research/reference_identity
    r1 = execute_research_template(
        "research/reference_identity",
        context={"unresolved_identities": ["UnknownAmbiguousMob"]},
    )
    assert r1["status"] == "BLOCKED"
    assert r1["proof"]["passed"] is False
    assert "Unresolved identities" in r1["proof"]["reason"]

    # 2. Failed checks block research/evidence_check and mark supported=False
    r2 = execute_research_template(
        "research/evidence_check",
        context={"failed_checks": ["Claim provenance cannot be traced to primary source"]},
    )
    assert r2["status"] == "BLOCKED"
    assert r2["proof"]["passed"] is False
    assert r2["output"]["supported"] is False
