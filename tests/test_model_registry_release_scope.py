from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.config_paths import config_path
from minecraft_mod_ai.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = config_path("model_registry.yaml")
LEGACY_REGISTRY_PATH = ROOT / "config" / "model_registry.yaml"
FOUNDATION_ROLES = ("planner", "researcher", "coder", "coder_safe")
RELEASE_FAMILIES = frozenset({"qwen3.5", "qwen3.8"})


def test_release_registry_has_one_packaged_canonical_source() -> None:
    expected = (ROOT / "minecraft_mod_ai" / "config" / "model_registry.yaml").resolve()
    assert REGISTRY_PATH == expected
    assert REGISTRY_PATH.is_file()
    assert not LEGACY_REGISTRY_PATH.exists()


def test_removed_foundation_paths_cannot_reenter_release_registry() -> None:
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "Qwen3.6" not in text
    assert "qwen3.6" not in text
    assert "remote_quality" not in text
    assert "Qwen3.8-9B" not in text


def test_every_real_release_profile_uses_one_qwen35_or_qwen38_foundation() -> None:
    registry = ModelRegistry(REGISTRY_PATH)
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
