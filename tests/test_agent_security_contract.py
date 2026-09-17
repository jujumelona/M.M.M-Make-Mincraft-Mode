from __future__ import annotations

from minecraft_mod_ai.model_router import _usable_rag_result


def test_rag_gate_accepts_concrete_legacy_evidence() -> None:
    assert _usable_rag_result({"hits": [{"text": "legacy evidence"}]})
    assert _usable_rag_result({"sources": [{"content": "official evidence"}]})


def test_rag_gate_rejects_observation_metadata_without_evidence() -> None:
    assert not _usable_rag_result(
        {
            "_mmm_observation": {
                "trust": "untrusted_data_only",
                "sanitized": True,
                "truncated": False,
            },
            "hint": "retry",
        }
    )


def test_rag_gate_rejects_contentless_positive_receipt() -> None:
    assert not _usable_rag_result(
        {
            "preserved_evidence": [
                {
                    "receipt": {
                        "result_count": 3,
                        "coverage_score": 0.55,
                        "relevance_score": 0.82,
                    }
                }
            ]
        }
    )


def test_rag_gate_accepts_concrete_evidence_with_positive_receipt() -> None:
    assert _usable_rag_result(
        {
            "hits": [{"text": "current project evidence"}],
            "receipt": {
                "result_count": 1,
                "coverage_score": 1.0,
                "relevance_score": 1.0,
            },
        }
    )


def test_rag_gate_rejects_nonpositive_or_malformed_receipt_scores() -> None:
    assert not _usable_rag_result(
        {
            "hits": [{"text": "evidence"}],
            "receipt": {
                "result_count": 2,
                "coverage_score": 0.0,
                "relevance_score": 0.8,
            },
        }
    )
    assert not _usable_rag_result(
        {
            "hits": [{"text": "evidence"}],
            "receipt": {
                "result_count": 2,
                "coverage_score": "not-a-number",
                "relevance_score": 0.8,
            },
        }
    )
    assert not _usable_rag_result(
        {
            "hits": [{"text": "evidence"}],
            "receipt": {
                "result_count": 2,
                "coverage_score": float("nan"),
                "relevance_score": 0.8,
            },
        }
    )
