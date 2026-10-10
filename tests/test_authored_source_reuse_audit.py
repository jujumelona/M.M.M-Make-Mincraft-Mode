from __future__ import annotations

import pytest
from minecraft_mod_ai.authored_reuse_bridge import audit_authored_source_reuse


def _plan():
    return {
        "schema_version": "mmm/grounded-repository-reuse-plan-v2",
        "capability_graph": {"nodes": ["block", "recipe"]},
        "capabilities": [
            {"capability": "block", "mode": "fresh", "source_id": ""},
            {"capability": "recipe", "mode": "fresh", "source_id": ""},
        ],
        "inspection_receipts": [{"status": "inspection_error"}],
        "proof_receipts": [{"status": "proof_error"}],
    }


def test_all_fresh_decisions_are_not_marked_reused():
    audit = audit_authored_source_reuse(_plan(), installed_donor_count=0)
    assert audit["status"] == "FRESH_IMPLEMENTATION_REQUIRED"
    assert audit["verified_transplant_count"] == 0
    assert audit["fresh_implementation_count"] == 2
    assert audit["proof_failure_count"] == 1


def test_missing_decision_is_rejected():
    data = _plan()
    data["capabilities"].pop()
    with pytest.raises(ValueError, match="UNCOVERED_CAPABILITIES"):
        audit_authored_source_reuse(data)


def test_duplicate_capability_is_rejected():
    data = _plan()
    data["capabilities"].append(dict(data["capabilities"][0]))
    with pytest.raises(ValueError, match="CAPABILITY_MISMATCH"):
        audit_authored_source_reuse(data)
