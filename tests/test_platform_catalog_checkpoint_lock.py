from __future__ import annotations

import json
import pytest

import minecraft_mod_ai.platform_catalog as catalog
from minecraft_mod_ai.platform_resolver import lock_from_adapter


def _adapter():
    return catalog.adapter_for_target("1.21.4", "fabric")


def _write_lock(path, adapter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_from_adapter(adapter)
    payload = {
        name: getattr(lock, name)
        for name in lock.__dataclass_fields__
    }
    path.write_text(
        json.dumps(payload),
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


def test_fabric_descriptor_identifies_loader_without_platform_lock(tmp_path, monkeypatch):
    adapter = _adapter()
    project_root = tmp_path / "project"
    resources = project_root / "src" / "main" / "resources"
    resources.mkdir(parents=True)
    (project_root / "gradle.properties").write_text(
        "minecraft_version=1.21.4\n",
        encoding="utf-8",
    )
    (resources / "fabric.mod.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "debugfixture",
                "version": "1.0.0",
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    def fake_adapter_for_target(version: str, loader: str):
        calls.append((version, loader))
        return adapter

    monkeypatch.setattr(catalog, "adapter_for_target", fake_adapter_for_target)

    resolved = catalog.adapter_from_project(project_root)

    assert resolved is adapter
    assert calls == [("1.21.4", "fabric")]


def test_malformed_fabric_descriptor_does_not_identify_loader(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    resources = project_root / "src" / "main" / "resources"
    resources.mkdir(parents=True)
    (project_root / "gradle.properties").write_text(
        "minecraft_version=1.21.4\n",
        encoding="utf-8",
    )
    (resources / "fabric.mod.json").write_text("{not-json", encoding="utf-8")

    monkeypatch.setattr(
        catalog,
        "adapter_for_target",
        lambda _version, _loader: pytest.fail("malformed descriptor must not select Fabric"),
    )

    with pytest.raises(
        ValueError,
        match="Existing project loader could not be identified unambiguously",
    ):
        catalog.adapter_from_project(project_root)


def test_incomplete_fabric_descriptor_does_not_identify_loader(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    resources = project_root / "src" / "main" / "resources"
    resources.mkdir(parents=True)
    (project_root / "gradle.properties").write_text(
        "minecraft_version=1.21.4\n",
        encoding="utf-8",
    )
    (resources / "fabric.mod.json").write_text(
        json.dumps({"schemaVersion": 1, "id": "debugfixture"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        catalog,
        "adapter_for_target",
        lambda _version, _loader: pytest.fail("incomplete descriptor must not select Fabric"),
    )

    with pytest.raises(
        ValueError,
        match="Existing project loader could not be identified unambiguously",
    ):
        catalog.adapter_from_project(project_root)
