from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.parametrize("owner", [
    "canonical_generators", "integrity_dispatcher", "request_runtime_ownership",
    "registered_leaf_binding", "host_generator_catalog_migration",
])
def test_feature_resume_invalidates_changed_runtime_implementation(monkeypatch, owner):
    revision = ["0" * 64]

    def digest(value):
        return revision[0] if getattr(value, "__name__", "") == "minecraft_mod_ai." + owner else "f" * 64

    monkeypatch.setattr(policy, "_file_digest", digest)
    before = policy.production_implementation_fingerprint("content")
    host_before = policy.production_implementation_fingerprint("host")
    revision[0] = "1" * 64
    assert policy.production_implementation_fingerprint("content") != before
    assert policy.production_implementation_fingerprint("host") == host_before
