from __future__ import annotations

import minecraft_mod_ai.complete_orchestrator as orchestrator_module
from minecraft_mod_ai import quality_evidence
from minecraft_mod_ai.atomic_playtest_evidence_contract import _matched_acceptance_refs


def test_only_matched_wait_for_results_count_as_atomic_evidence() -> None:
    receipt = {
        "status": "PASS",
        "results": [
            {
                "action": "wait_for",
                "params": {"acceptance_ref": "acceptance:00000000"},
                "result": {"matched": True},
            },
            {
                "action": "wait_for",
                "params": {"acceptance_ref": "acceptance:00000001"},
                "result": {"matched": False},
            },
            {
                "action": "status",
                "params": {"acceptance_ref": "acceptance:00000002"},
                "result": {"connected": True},
            },
            {
                "action": "wait_for",
                "params": {
                    "acceptance_refs": [
                        "acceptance:00000003",
                        "not-an-acceptance",
                    ]
                },
                "result": {"matched": True},
            },
        ],
    }
    assert _matched_acceptance_refs(receipt) == {
        "acceptance:00000000",
        "acceptance:00000003",
    }


def test_atomic_playtest_verifier_is_in_real_quality_path() -> None:
    assert getattr(
        quality_evidence.compile_quality_evidence,
        "_mmm_atomic_playtest_evidence",
        False,
    )
    assert (
        orchestrator_module.compile_quality_evidence
        is quality_evidence.compile_quality_evidence
    )
