from __future__ import annotations

from minecraft_mod_ai import custom_module_generator
from minecraft_mod_ai import progress_aware_tool_loop as tool_loop


def test_runtime_finalization_installs_direct_task_authority() -> None:
    assert getattr(tool_loop, "_mmm_direct_task_mutation_authority_v1", False) is True
    assert getattr(custom_module_generator, "_mmm_direct_task_mutation_authority_v1", False) is True


def test_runtime_finalization_installs_final_mutation_guard() -> None:
    assert getattr(tool_loop, "_mmm_mutation_authority_final_guard_v1", False) is True
