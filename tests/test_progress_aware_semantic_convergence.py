from __future__ import annotations

from minecraft_mod_ai.progress_aware_tool_loop import HostRunState


def test_semantic_state_recurs_without_numeric_retry_threshold():
    state = HostRunState()
    semantic = {
        "phase": "ACT",
        "target": "DebugToken.java",
        "verifier_fingerprint": "diag-a",
        "tool_results": [{"name": "apply_source_edit", "failure_code": "MUTATION_UNCHANGED"}],
    }
    assert state.record_no_progress_result(semantic) is False
    assert state.semantic_fixed_point is False
    assert state.record_no_progress_result(semantic) is True
    assert state.semantic_fixed_point is True


def test_new_semantic_frontier_is_not_convergence():
    state = HostRunState()
    first = {"phase": "ACT", "verifier_fingerprint": "diag-a", "tool_results": [{"name": "apply_source_edit", "failure_code": "A"}]}
    second = {"phase": "ACT", "verifier_fingerprint": "diag-b", "tool_results": [{"name": "apply_source_edit", "failure_code": "A"}]}
    assert state.record_no_progress_result(first) is False
    assert state.record_no_progress_result(second) is False
    assert state.semantic_fixed_point is False


def test_material_progress_clears_semantic_recurrence_memory():
    state = HostRunState()
    semantic = {"phase": "ACT", "verifier_fingerprint": "diag-a", "tool_results": [{"name": "apply_source_edit", "failure_code": "A"}]}
    assert state.record_no_progress_result(semantic) is False
    assert state.record_no_progress_result(semantic) is True
    state.clear_no_progress_result()
    assert state.semantic_fixed_point is False
    assert state.record_no_progress_result(semantic) is False


def test_semantic_cycle_detection_catches_nonconsecutive_recurrence():
    state = HostRunState()
    a = {"phase": "OBSERVE", "tool_results": [{"name": "search_code_rag", "failure_code": "EMPTY"}]}
    b = {"phase": "OBSERVE", "tool_results": [{"name": "java_workspace_symbols", "failure_code": "EMPTY"}]}
    assert state.record_no_progress_result(a) is False
    assert state.record_no_progress_result(b) is False
    assert state.record_no_progress_result(a) is True


def test_fixed_point_records_exact_first_and_repeat_steps():
    state = HostRunState()
    semantic = {
        "phase": "OBSERVE",
        "calls": [
            {"name": "external_mcp_schema", "capability": "source_search"}
        ],
        "results": [{"name": "external_mcp_schema", "ok": True}],
    }
    state.step_index = 4
    assert state.record_no_progress_result(semantic) is False
    state.step_index = 9
    assert state.record_no_progress_result(semantic) is True
    assert state.fixed_point_first_seen_step == 4
    assert state.fixed_point_repeat_step == 9
    assert state.fixed_point_digest
    assert state.fixed_point_snapshot["phase"] == "OBSERVE"
