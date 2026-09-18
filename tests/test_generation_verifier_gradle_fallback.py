from __future__ import annotations

import inspect

from minecraft_mod_ai import generation_verifier_resilience
from minecraft_mod_ai import progress_aware_tool_loop


def test_generation_verifier_has_no_hidden_gradle_fallback_or_corroboration() -> None:
    source = inspect.getsource(generation_verifier_resilience)
    assert "generation_verifier_fallback_installation" not in source
    assert "_run_gradle_fallback" not in source
    assert "_run_gradle_corroboration" not in source
    assert "run_gradle_build" not in source
    assert not getattr(
        generation_verifier_resilience.run_generation_verifier,
        "_mmm_generation_gradle_fallback",
        False,
    )


def test_gradle_build_is_not_a_generation_verifier_surface() -> None:
    assert "run_gradle_build" not in progress_aware_tool_loop._VERIFY_TOOLS
    assert "gradle_build" not in progress_aware_tool_loop._VERIFY_TOOLS
