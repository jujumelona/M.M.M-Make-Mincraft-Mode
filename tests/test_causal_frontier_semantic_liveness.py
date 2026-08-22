from __future__ import annotations

import pytest

from minecraft_mod_ai.causal_frontier_adapter import CausalFrontierAdapter
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
