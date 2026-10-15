"""Guard the production regression: 484405 API candidates, 241 reads, zero nodes."""
from __future__ import annotations

from minecraft_mod_ai.construction_operation_choices import (
    MAX_OPERATION_EVIDENCE_READS,
    operation_context_branches,
    operation_next_schema,
)
from minecraft_mod_ai.construction_state import ConstructionState


def _state():
    state = ConstructionState({}, "example.Owner", "src/main/java/example/Owner.java")
    state.enter("operation_graph")
    state.selection_goal = {
        "node_id": "operation_0",
        "purpose": "Register an entity",
        "excluded_operations": [],
    }
    return state


def _actions(branches):
    return {
        action
        for branch in branches
        for action in (
            [branch["properties"]["action"]["const"]]
            if "const" in branch["properties"].get("action", {})
            else branch["properties"].get("action", {}).get("enum", [])
        )
    }


def test_operation_selection_does_not_page_global_symbols():
    state = _state()
    branches = operation_context_branches(
        state, nodes=[], source_units={}, pinned_source="", api_resolution=False,
    )
    assert not _actions(branches) & {"symbols", "more_symbols", "scopes"}
    # API navigation is allowed only once a typed operation was selected.
    bind_branches = operation_context_branches(
        state, nodes=[], source_units={}, pinned_source="", api_resolution=True,
    )
    assert "symbols" in _actions(bind_branches)


def test_operation_evidence_has_finite_no_progress_window():
    state = _state()
    for offset in range(MAX_OPERATION_EVIDENCE_READS):
        state.requests.add(("symbols", f"symbol_{offset}", 0))
    assert operation_context_branches(
        state, nodes=[], source_units={}, pinned_source="", api_resolution=True,
    ) == []


def test_reconsider_clears_search_window_without_losing_slot_exclusions():
    state = _state()
    state.requests.add(("symbols", "example", 0))
    state.symbol_goal = "old_query"
    state.selection_goal["excluded_operations"].append("call")
    state.operation_goal = {
        "action": "call", "node_id": "operation_0", "purpose": "Register an entity",
    }
    state.reconsider_operation()
    assert not state.requests
    assert state.symbol_goal is None
    assert state.operation_goal is None
    assert state.selection_goal["excluded_operations"] == ["call"]
    restored = _state()
    restored.restore(state.export())
    assert restored.selection_goal == state.selection_goal


def test_rejected_operation_not_offered_again_for_same_slot():
    state = _state()
    state.selection_goal["excluded_operations"] = ["call", "new"]
    member = {
        "kind": "METHOD", "owner": "example.Owner",
        "name": "registerAlien", "return_type": "void",
        "static": True, "parameters": [],
    }
    schema = operation_next_schema(state, nodes=[], declaration=member, can_complete=False)
    actions = _actions(schema["oneOf"])
    assert "call" not in actions
    assert "new" not in actions
    assert "return" in actions


def test_memory_navigation_also_exhausts_same_no_progress_window():
    state = _state()
    state.operation_reads = MAX_OPERATION_EVIDENCE_READS
    assert not state.requests
    assert operation_context_branches(
        state, nodes=[], source_units={}, pinned_source="", api_resolution=True,
    ) == []
    saved = state.export()
    restored = _state()
    restored.restore(saved)
    assert restored.operation_reads == MAX_OPERATION_EVIDENCE_READS
    restored.reconsider_operation()
    assert restored.operation_reads == 0
