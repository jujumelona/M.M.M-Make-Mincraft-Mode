from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATHS = (
    ROOT / "config" / "model_registry.yaml",
    ROOT / "minecraft_mod_ai" / "config" / "model_registry.yaml",
)
FOUNDATION_ROLES = ("planner", "researcher", "coder", "coder_safe")
RELEASE_FAMILIES = frozenset({"qwen3.5", "qwen3.8"})


def test_checkout_and_packaged_registry_are_identical() -> None:
    assert REGISTRY_PATHS[0].read_bytes() == REGISTRY_PATHS[1].read_bytes()


def test_removed_foundation_paths_cannot_reenter_release_registry() -> None:
    for path in REGISTRY_PATHS:
        text = path.read_text(encoding="utf-8")
        assert "Qwen3.6" not in text
        assert "qwen3.6" not in text
        assert "remote_quality" not in text
        assert "Qwen3.8-9B" not in text


def test_every_real_release_profile_uses_one_qwen35_or_qwen38_foundation() -> None:
    registry = ModelRegistry(REGISTRY_PATHS[1])
    for name in registry.profile_names():
        profile = registry.load_profile(name)
        configs = tuple(profile.roles[role] for role in FOUNDATION_ROLES)
        if all(config.adapter == "mock" for config in configs):
            # Unit-only deterministic fixture; it must never be treated as real-model evidence.
            continue
        identities = {
            (config.provider, config.adapter, config.model_id, config.base_url)
            for config in configs
        }
        assert len(identities) == 1, name
        families = {str(config.extra.get("qwen_family", "")) for config in configs}
        assert families <= RELEASE_FAMILIES
        assert len(families) == 1, name
