from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.causal_frontier_adapter import (
    CausalFrontierAdapter,
    _latest_tool_result_signature,
)
from minecraft_mod_ai.model_adapters import ModelConfigurationError


class _Inner:
    pass


def _adapter() -> CausalFrontierAdapter:
    return CausalFrontierAdapter(
        _Inner(),
        stage="generation",
        role="coder",
        require_fresh_evidence=True,
    )


def test_semantically_identical_retrieval_frontier_fails_before_hard_round_guard() -> None:
    adapter = _adapter()
    kwargs = {
        "state": frozenset(
            {"project_observed", "workspace_bound"}
        ),
        "goals": ("repair",),
        "names": ("search_project_rag", "java_workspace_symbols", "search_code_rag"),
    }

    adapter._guard_semantic_progress(**kwargs)
    adapter._guard_semantic_progress(**kwargs)
    with pytest.raises(ModelConfigurationError, match="semantic no-progress fixed point"):
        adapter._guard_semantic_progress(**kwargs)


def test_new_causal_state_resets_semantic_stall_counter() -> None:
    adapter = _adapter()
    adapter._guard_semantic_progress(
        state=frozenset({"workspace_bound"}),
        goals=("repair",),
        names=("search_code_rag",),
    )
    adapter._guard_semantic_progress(
        state=frozenset({"workspace_bound"}),
        goals=("repair",),
        names=("search_code_rag",),
    )
    adapter._guard_semantic_progress(
        state=frozenset({"workspace_bound", "code_evidence", "evidence_ready"}),
        goals=("repair",),
        names=("java_workspace_symbols",),
    )
    adapter._guard_semantic_progress(
        state=frozenset({"workspace_bound", "code_evidence", "evidence_ready"}),
        goals=("repair",),
        names=("java_workspace_symbols",),
    )


def test_novel_result_resets_stall_but_query_and_telemetry_noise_do_not() -> None:
    def transcript(*, query: str, path: str, score: float) -> tuple[dict[str, object], ...]:
        return (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call-{query}",
                        "type": "function",
                        "function": {
                            "name": "search_code_rag",
                            "arguments": json.dumps({"query": query}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": f"call-{query}",
                "name": "search_code_rag",
                "content": json.dumps(
                    {
                        "ok": True,
                        "tool": "search_code_rag",
                        "result": {
                            "query": query,
                            "score": score,
                            "elapsed_ms": score * 100,
                            "hits": [{"path": path, "line": 10}],
                        },
                    }
                ),
            },
        )

    first = _latest_tool_result_signature(
        transcript(query="first wording", path="A.java", score=0.80)
    )
    paraphrase = _latest_tool_result_signature(
        transcript(query="different wording", path="A.java", score=0.95)
    )
    novel = _latest_tool_result_signature(
        transcript(query="third wording", path="B.java", score=0.95)
    )

    assert first == paraphrase
    assert first != novel

    adapter = _adapter()
    kwargs = {
        "state": frozenset({"workspace_bound", "project_observed", "code_evidence"}),
        "goals": ("repair",),
        "names": ("search_code_rag",),
    }
    adapter._guard_semantic_progress(**kwargs, result_signature=first)
    adapter._guard_semantic_progress(**kwargs, result_signature=first)
    adapter._guard_semantic_progress(**kwargs, result_signature=novel)
    adapter._guard_semantic_progress(**kwargs, result_signature=novel)


def test_mutation_frontier_remains_owned_by_existing_mutation_retry_policy() -> None:
    adapter = _adapter()
    for _ in range(8):
        adapter._guard_semantic_progress(
            state=frozenset(
                {"workspace_bound", "project_observed", "code_evidence", "evidence_ready"}
            ),
            goals=("repair",),
            names=("apply_source_edit",),
        )


def test_empty_or_final_frontier_does_not_create_semantic_stall() -> None:
    adapter = _adapter()
    for _ in range(4):
        adapter._guard_semantic_progress(
            state=frozenset(),
            goals=(),
            names=(),
        )
