from __future__ import annotations

from minecraft_mod_ai.retrieval_progress import (
    RetrievalDecision,
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


def test_source_mutation_names_are_canonical_and_shared() -> None:
    assert "apply_source_patch" in SOURCE_MUTATION_NAMES
    assert "apply_source_edit" in SOURCE_MUTATION_NAMES
