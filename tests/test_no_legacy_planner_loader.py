from pathlib import Path

import pytest

from minecraft_mod_ai.planner import HeuristicPlanner, LocalTransformersPlanner
from minecraft_mod_ai.spec import SpecValidationError


def test_planner_source_contains_no_invalid_qwen_id_or_silent_fallback() -> None:
    source = Path("minecraft_mod_ai/planner.py").read_text(encoding="utf-8")
    for forbidden in (
        "Qwen/Qwen3.5-9B-Instruct",
        'self.last_backend = "deterministic-fallback"',
        "return self.fallback.plan(prompt)",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
    ):
        assert forbidden not in source


def test_compatibility_planner_rejects_direct_overrides_and_fallbacks() -> None:
    with pytest.raises(SpecValidationError, match="Direct model_id overrides"):
        LocalTransformersPlanner(model_id="Qwen/Qwen3.5-4B")
    with pytest.raises(SpecValidationError, match="fallback is disabled"):
        LocalTransformersPlanner(fallback=HeuristicPlanner())
    with pytest.raises(SpecValidationError, match="max_new_tokens overrides"):
        LocalTransformersPlanner(max_new_tokens=100)


def test_compatibility_planner_routes_only_through_named_profile() -> None:
    planner = LocalTransformersPlanner(profile="t4_local")
    assert planner.profile == "t4_local"
    assert planner.last_backend == "role-router:t4_local"
