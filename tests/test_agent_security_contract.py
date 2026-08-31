from __future__ import annotations

from minecraft_mod_ai.agent_security_contract import usable_rag_result


def test_rag_gate_accepts_positive_receipt_even_when_hits_were_truncated() -> None:
    assert usable_rag_result(
        {
            "_mmm_observation": {
                "trust": "untrusted_data_only",
                "sanitized": True,
                "truncated": True,
            },
            "preserved_evidence": [
                {
                    "receipt": {
                        "result_count": 3,
                        "coverage_score": 0.55,
                        "relevance_score": 0.82,
                        "warnings": ["coverage_below_route_threshold"],
                    }
                }
            ],
        }
    )


def test_rag_gate_rejects_observation_metadata_without_evidence() -> None:
    assert not usable_rag_result(
        {
            "_mmm_observation": {
                "trust": "untrusted_data_only",
                "sanitized": True,
                "truncated": False,
            },
            "hint": "retry",
        }
    )


def test_rag_gate_fails_closed_on_required_metadata_mismatch() -> None:
    assert not usable_rag_result(
        {
            "hits": [{"text": "stale hit"}],
            "receipt": {
                "result_count": 1,
                "coverage_score": 1.0,
                "relevance_score": 1.0,
                "warnings": ["required_metadata_mismatch"],
            },
        }
    )


def test_rag_gate_rejects_malformed_or_nonfinite_scores() -> None:
    assert not usable_rag_result(
        {
            "receipt": {
                "result_count": 2,
                "coverage_score": "not-a-number",
                "relevance_score": 0.8,
            }
        }
    )
    assert not usable_rag_result(
        {
            "receipt": {
                "result_count": 2,
                "coverage_score": float("nan"),
                "relevance_score": 0.8,
            }
        }
    )


def test_rag_gate_keeps_legacy_known_evidence_pack_compatibility() -> None:
    assert usable_rag_result({"hits": [{"text": "legacy evidence"}]})
    assert usable_rag_result({"sources": [{"source_id": "official"}]})
    assert not usable_rag_result({"status": "searched"})


def test_rag_gate_rejects_receipt_and_evidence_from_different_siblings() -> None:
    assert not usable_rag_result(
        {
            "results": [
                {
                    "receipt": {
                        "result_count": 2,
                        "coverage_score": 0.0,
                        "relevance_score": 0.0,
                    }
                },
                {"hits": [{"text": "unreceipted sibling evidence"}]},
            ]
        }
    )


def test_rag_gate_accepts_positive_receipt_with_same_scope_legacy_evidence() -> None:
    assert usable_rag_result(
        {
            "results": [
                {
                    "receipt": {
                        "result_count": 2,
                        "coverage_score": 0.0,
                        "relevance_score": 0.0,
                    },
                    "payload": {
                        "hits": [{"text": "same result evidence"}],
                    },
                }
            ]
        }
    )


def test_rag_gate_does_not_rescue_terminal_receipt_with_sibling_hits() -> None:
    assert not usable_rag_result(
        {
            "results": [
                {
                    "receipt": {
                        "result_count": 1,
                        "coverage_score": 1.0,
                        "relevance_score": 1.0,
                        "warnings": ["required_metadata_mismatch"],
                    }
                },
                {"hits": [{"text": "stale sibling hit"}]},
            ]
        }
    )
