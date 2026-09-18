from __future__ import annotations

from minecraft_mod_ai.progress_aware_tool_loop import (
    _MUTATION_ACT_TOOLS,
    _READ_OBSERVE_TOOLS,
    _VERIFY_TOOLS,
    LoopPhase,
)


def test_host_tool_phase_classification_is_canonical() -> None:
    assert "search_code_rag" in _READ_OBSERVE_TOOLS
    assert "search_project_rag" in _READ_OBSERVE_TOOLS
    assert "external_mcp_call" in _READ_OBSERVE_TOOLS
    assert "java_workspace_symbols" in _READ_OBSERVE_TOOLS

    assert "apply_source_edit" in _MUTATION_ACT_TOOLS
    assert "apply_source_patch" in _MUTATION_ACT_TOOLS
    assert "apply_java_operations" in _MUTATION_ACT_TOOLS

    assert "java_diagnostics" in _VERIFY_TOOLS
    assert "run_gradle_build" not in _VERIFY_TOOLS
    assert "run_gametest" in _VERIFY_TOOLS


def test_mutation_tool_set_is_disjoint_from_observe_and_verify() -> None:
    assert _MUTATION_ACT_TOOLS.isdisjoint(_READ_OBSERVE_TOOLS)
    assert _MUTATION_ACT_TOOLS.isdisjoint(_VERIFY_TOOLS)


def test_loop_phase_values_cover_all_execution_phases() -> None:
    phases = {p.value for p in LoopPhase}
    assert "OBSERVE" in phases
    assert "ACT" in phases
    assert "VERIFY" in phases
    assert "RECOVER" in phases
