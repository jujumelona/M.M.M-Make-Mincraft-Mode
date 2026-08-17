from __future__ import annotations

from minecraft_mod_ai.model_registry import ModelRegistry


def test_all_registered_local_llama_roles_are_explicit_exclusive_gpu_consumers() -> None:
    registry = ModelRegistry()
    profiles = [
        profile
        for profile, raw_profile in registry._raw_profiles.items()
        if isinstance(raw_profile, dict)
        and isinstance(raw_profile.get("roles"), dict)
        and isinstance(raw_profile["roles"].get("planner"), dict)
        and raw_profile["roles"]["planner"].get("adapter") == "llama_cpp"
    ]
    assert profiles

    for profile in profiles:
        for role in ("planner", "researcher", "coder", "coder_safe", "visual_critic"):
            config = registry.role(profile, role)
            assert config.provider == "local"
            assert config.adapter == "llama_cpp"
            assert config.exclusive_gpu is True


def test_fast_mock_does_not_gain_gpu_ownership() -> None:
    config = ModelRegistry().role("fast_test", "planner")
    assert config.adapter == "mock"
    assert config.exclusive_gpu is False
