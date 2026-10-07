from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import production_checkpoint_policy as policy


def _fingerprinted_names(monkeypatch, stage: str) -> set[str]:
    names: set[str] = set()

    def record(value):
        if isinstance(value, Path):
            names.add(value.name)
        else:
            names.add(str(getattr(value, "__name__", "")))
        return "0" * 64

    monkeypatch.setattr(policy, "_file_digest", record)
    fingerprint = policy.production_implementation_fingerprint(stage)
    assert fingerprint.startswith("sha256:")
    assert len(fingerprint) == 71
    return names


def test_asset_resume_fingerprint_includes_model_and_image_pipeline(monkeypatch) -> None:
    names = _fingerprinted_names(monkeypatch, "assets")
    assert "model_registry.yaml" in names
    assert "minecraft_mod_ai.model_registry" in names
    assert "minecraft_mod_ai.resource_asset_production" in names
    assert "minecraft_mod_ai.resource_image_pipeline" in names
    assert "minecraft_mod_ai.model_adapters.image_diffusion" in names
    assert "minecraft_mod_ai.production_routing_contract" in names


def test_native_stage_fingerprints_include_their_owner_generators(monkeypatch) -> None:
    assert "minecraft_mod_ai.extended_content_generator" in _fingerprinted_names(
        monkeypatch, "content"
    )
    assert "minecraft_mod_ai.system_pack_generator" in _fingerprinted_names(
        monkeypatch, "system"
    )
    assert "minecraft_mod_ai.geckolib_generator" in _fingerprinted_names(
        monkeypatch, "entity"
    )
    assert "minecraft_mod_ai.typed_plan_production" in _fingerprinted_names(
        monkeypatch, "host"
    )
