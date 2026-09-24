from __future__ import annotations

import inspect

from minecraft_mod_ai import generation_loop_outcomes
from minecraft_mod_ai import progress_aware_tool_loop as tool_loop


def test_mutation_authority_uses_canonical_progress_and_outcome_owners() -> None:
    assert inspect.getmodule(tool_loop.is_mutation_ready) is tool_loop
    assert inspect.getmodule(tool_loop._verification_outcome) is generation_loop_outcomes
    assert tool_loop._authority_allowed("developer", {}) is True
    assert tool_loop._authority_allowed("user", {}) is False


def test_final_mutation_guard_is_source_owned_not_runtime_installed() -> None:
    assert "apply_source_edit" in tool_loop._MUTATION_ACT_TOOLS
    assert "repair_project" in tool_loop._MUTATION_ACT_TOOLS
    assert getattr(tool_loop, "_mmm_mutation_authority_final_guard_v1", False) is True
