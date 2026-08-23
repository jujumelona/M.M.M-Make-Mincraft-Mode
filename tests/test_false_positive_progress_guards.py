from __future__ import annotations

from minecraft_mod_ai.causal_frontier_adapter import CausalFrontierAdapter
from minecraft_mod_ai.coder_tool_route_integrity_contract import _preferred_visible_mutation
from minecraft_mod_ai.forced_tool_execution_contract import _SOURCE_MUTATION_TOOLS
from minecraft_mod_ai.retrieval_progress import (
    RetrievalDecision,
    RetrievalObservation,
    RetrievalProgress,
)
from minecraft_mod_ai.source_mutation_contract import SOURCE_MUTATION_NAMES


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


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

    # Repeated evidence from one query must not permanently disable the source.
    assert progress.begin("search_code_rag", third) is RetrievalDecision.EXECUTE
    assert progress.begin("search_code_rag", third) is RetrievalDecision.DUPLICATE_QUERY


def test_writable_adapter_has_no_independent_mutation_retry_budget() -> None:
    tools = (_tool("apply_source_patch"), _tool("apply_source_edit"))

    # Retry/evidence-epoch state belongs to CausalStateLedger. This helper only
    # chooses among mutation actions already exposed by the current causal frontier.
    assert _preferred_visible_mutation(tools) == "apply_source_patch"


def test_forced_tool_contract_uses_canonical_mutation_names() -> None:
    assert _SOURCE_MUTATION_TOOLS is SOURCE_MUTATION_NAMES
    assert "apply_source_patch" in _SOURCE_MUTATION_TOOLS


def test_causal_frontier_has_no_fuzzy_semantic_stall_guard() -> None:
    assert not hasattr(CausalFrontierAdapter, "_guard_semantic_progress")
