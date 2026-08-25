from __future__ import annotations

import pytest

from minecraft_mod_ai.retrieval_progress import (
    RetrievalDecision,
    RetrievalNoProgressError,
    RetrievalObservation,
    RetrievalProgress,
)
from minecraft_mod_ai.source_mutation_contract import SOURCE_MUTATION_NAMES


def test_duplicate_evidence_does_not_blacklist_retrieval_source() -> None:
    progress = RetrievalProgress()
    first = {"query": "alpha symbol"}
    second = {"query": "beta symbol"}
    third = {"query": "gamma symbol"}
    evidence = {"files": ["src/Main.java"]}

    assert progress.begin("search_code_rag", first) is RetrievalDecision.EXECUTE
    assert (
        progress.observe("search_code_rag", first, evidence, usable=True)
        is RetrievalObservation.FRESH
    )
    assert progress.begin("search_code_rag", second) is RetrievalDecision.EXECUTE
    assert (
        progress.observe("search_code_rag", second, evidence, usable=True)
        is RetrievalObservation.DUPLICATE_EVIDENCE
    )
    assert progress.begin("search_code_rag", third) is RetrievalDecision.EXECUTE
    assert progress.begin("search_code_rag", third) is RetrievalDecision.DUPLICATE_QUERY


def test_distinct_weak_queries_are_bounded_before_context_growth() -> None:
    progress = RetrievalProgress(no_progress_limit=3)

    for index in range(2):
        arguments = {"query": f"missing symbol {index}"}
        assert progress.begin("search_code_rag", arguments) is RetrievalDecision.EXECUTE
        assert (
            progress.observe("search_code_rag", arguments, {"hits": []}, usable=False)
            is RetrievalObservation.WEAK
        )

    final_arguments = {"query": "missing symbol final"}
    assert progress.begin("search_code_rag", final_arguments) is RetrievalDecision.EXECUTE
    with pytest.raises(RetrievalNoProgressError, match="no novel usable evidence"):
        progress.observe(
            "search_code_rag",
            final_arguments,
            {"hits": []},
            usable=False,
        )


def test_fresh_evidence_resets_weak_retrieval_streak() -> None:
    progress = RetrievalProgress(no_progress_limit=3)

    for index in range(2):
        arguments = {"query": f"weak before fresh {index}"}
        assert (
            progress.observe("search_code_rag", arguments, {"hits": []}, usable=False)
            is RetrievalObservation.WEAK
        )

    fresh_arguments = {"query": "real symbol"}
    assert (
        progress.observe(
            "search_code_rag",
            fresh_arguments,
            {"hits": [{"path": "src/Main.java", "line": 1}]},
            usable=True,
        )
        is RetrievalObservation.FRESH
    )
    assert progress.no_progress_observations == 0

    for index in range(2):
        arguments = {"query": f"weak after fresh {index}"}
        assert (
            progress.observe("search_code_rag", arguments, {"hits": []}, usable=False)
            is RetrievalObservation.WEAK
        )


def test_source_mutation_names_are_canonical_and_shared() -> None:
    assert "apply_source_patch" in SOURCE_MUTATION_NAMES
    assert "apply_source_edit" in SOURCE_MUTATION_NAMES
