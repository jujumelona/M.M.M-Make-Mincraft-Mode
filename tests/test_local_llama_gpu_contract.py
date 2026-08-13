from __future__ import annotations

import pytest

from minecraft_mod_ai.llama_parallel_runtime_contract import _planner_parallel_capacity
from minecraft_mod_ai.model_registry import ModelRegistry


_LOCAL_LLAMA_PROFILES = (
    "mini_mod",
    "Qwen3.6-35B_23GB",
    "Qwen3.6-27B_18GB",
    "Qwen3.6-27B_14GB",
    "Gemma4-26B_14GB",
    "Gemma4-12B_7GB",
    "Qwen3.5-9B_6GB",
    "t4_local",
    "t4_quality",
)


@pytest.mark.parametrize("profile", _LOCAL_LLAMA_PROFILES)
def test_local_llama_roles_are_explicit_exclusive_gpu_consumers(profile: str) -> None:
    registry = ModelRegistry()
    for role in ("planner", "researcher", "coder", "coder_safe", "visual_critic"):
        config = registry.role(profile, role)
        assert config.provider == "local"
        assert config.adapter == "llama_cpp"
        assert config.exclusive_gpu is True


def test_exclusive_local_llama_enables_selected_parallel_read_capacity(monkeypatch) -> None:
    monkeypatch.setenv("MMM_LLAMA_ACTIVE_PARALLEL", "3")

    class Router:
        registry = ModelRegistry()
        profile = "Qwen3.5-9B_6GB"

    assert _planner_parallel_capacity(Router(), 4) == 3


def test_fast_mock_does_not_gain_gpu_ownership() -> None:
    config = ModelRegistry().role("fast_test", "planner")
    assert config.adapter == "mock"
    assert config.exclusive_gpu is False
