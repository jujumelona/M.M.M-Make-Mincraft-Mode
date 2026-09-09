from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import minecraft_mod_ai.platform_catalog as catalog


def _adapter() -> SimpleNamespace:
    return SimpleNamespace(
        minecraft_version="1.21.4",
        loader="fabric",
        java_version=21,
        fabric_loader="0.16.10",
        fabric_api="0.119.2+1.21.4",
        fabric_loom="1.9.2",
        gradle="8.10.2",
        mappings_applicable=False,
        mappings_kind="native",
        mappings_version="",
    )


def _write_lock(path, adapter: SimpleNamespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "minecraft_version": adapter.minecraft_version,
                "loader": adapter.loader,
                "java_version": adapter.java_version,
                "fabric_loader": adapter.fabric_loader,
                "fabric_api": adapter.fabric_api,
                "fabric_loom": adapter.fabric_loom,
                "gradle": adapter.gradle,
            }
        ),
        encoding="utf-8",
    )


def test_checkpoint_project_inherits_authoritative_platform_lock(tmp_path, monkeypatch):
    adapter = _adapter()
    output_root = tmp_path / "complete-colab-run"
    metadata_root = output_root / ".minecraft_ai"
    checkpoint_root = (
        metadata_root
        / ".mmm-custom-checkpoints"
        / ("b" * 64)
        / "project"
    )
    checkpoint_root.mkdir(parents=True)
    _write_lock(metadata_root / "platform-lock.json", adapter)

    calls: list[tuple[str, str]] = []

    def fake_adapter_for_target(version: str, loader: str):
        calls.append((version, loader))
        return adapter

    monkeypatch.setattr(catalog, "adapter_for_target", fake_adapter_for_target)

    resolved = catalog.adapter_from_project(checkpoint_root)

    assert resolved is adapter
    assert calls == [("1.21.4", "fabric")]


def test_similar_non_checkpoint_path_does_not_inherit_parent_lock(tmp_path, monkeypatch):
    adapter = _adapter()
    metadata_root = tmp_path / ".minecraft_ai"
    project_root = metadata_root / "not-checkpoints" / ("c" * 64) / "project"
    project_root.mkdir(parents=True)
    (project_root / "gradle.properties").write_text("", encoding="utf-8")
    _write_lock(metadata_root / "platform-lock.json", adapter)

    monkeypatch.setattr(
        catalog,
        "adapter_for_target",
        lambda _version, _loader: pytest.fail("ancestor lock must not be trusted"),
    )

    with pytest.raises(
        ValueError,
        match="Existing project loader could not be identified unambiguously",
    ):
        catalog.adapter_from_project(project_root)


def test_checkpoint_with_non_sha_key_does_not_inherit_parent_lock(tmp_path, monkeypatch):
    adapter = _adapter()
    metadata_root = tmp_path / ".minecraft_ai"
    project_root = (
        metadata_root / ".mmm-custom-checkpoints" / "not-a-sha256" / "project"
    )
    project_root.mkdir(parents=True)
    (project_root / "gradle.properties").write_text("", encoding="utf-8")
    _write_lock(metadata_root / "platform-lock.json", adapter)

    monkeypatch.setattr(
        catalog,
        "adapter_for_target",
        lambda _version, _loader: pytest.fail("invalid checkpoint key must not inherit lock"),
    )

    with pytest.raises(
        ValueError,
        match="Existing project loader could not be identified unambiguously",
    ):
        catalog.adapter_from_project(project_root)
