from __future__ import annotations

from minecraft_mod_ai import model_router


def test_positive_receipt_hits_survive_zero_reranker_scores() -> None:
    value = {
        "receipt": {
            "result_count": 2,
            "coverage_score": 0.0,
            "relevance_score": 0.0,
        },
        "hits": [
            {"path": "src/main/java/Example.java", "line": 7},
            {"path": "src/main/java/Other.java", "line": 12},
        ],
    }

    assert model_router._usable_rag_result(value) is True


def test_zero_result_receipt_does_not_accept_stale_hits() -> None:
    value = {
        "receipt": {
            "result_count": 0,
            "coverage_score": 0.0,
            "relevance_score": 0.0,
        },
        "hits": [{"path": "stale.java", "line": 1}],
    }

    assert model_router._usable_rag_result(value) is False


def test_scored_receipt_remains_usable_without_hits() -> None:
    value = {
        "receipt": {
            "result_count": 1,
            "coverage_score": 0.5,
            "relevance_score": 0.7,
        }
    }

    assert model_router._usable_rag_result(value) is True
