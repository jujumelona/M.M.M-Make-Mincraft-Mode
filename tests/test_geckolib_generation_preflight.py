from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import geckolib_generation_contract as contract
from minecraft_mod_ai import geckolib_generator
from minecraft_mod_ai.generation_boundary_reconciliation import (
    _install_geckolib_project_preflight,
)


def _project_info(
    tmp_path: Path,
    *,
    depends: object = None,
    client: object = None,
    build_text: str = "plugins {}\n\ndependencies {\n}\n",
) -> SimpleNamespace:
    metadata = tmp_path / "src/main/resources/fabric.mod.json"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    entrypoints: dict[str, object] = {"main": ["com.example.TestMod"]}
    if client is not None:
        entrypoints["client"] = client
    payload: dict[str, object] = {
        "id": "example",
        "entrypoints": entrypoints,
    }
    if depends is not None:
        payload["depends"] = depends
    metadata.write_text(json.dumps(payload), encoding="utf-8")

    main_java = tmp_path / "src/main/java/com/example/TestMod.java"
    main_java.parent.mkdir(parents=True, exist_ok=True)
    main_java.write_text(
        "package com.example; public final class TestMod {}\n",
        encoding="utf-8",
    )
    (tmp_path / "build.gradle").write_text(build_text, encoding="utf-8")
    return SimpleNamespace(
        root=tmp_path,
        mod_id="example",
        package_name="com.example",
        main_java=main_java,
        fabric_mod_json=metadata,
    )


def _adapter() -> SimpleNamespace:
    return SimpleNamespace(loader="fabric", minecraft_version="1.21.8")


def test_preflight_rejects_invalid_fabric_metadata_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "adapter_from_project", lambda _root: _adapter())

    malformed_depends = _project_info(tmp_path / "depends", depends=[])
    with pytest.raises(contract.GeckoLibGenerationContractError, match="depends must be an object"):
        contract.validate_geckolib_project_preflight(malformed_depends)

    malformed_client = _project_info(tmp_path / "client", client="com.example.Client")
    with pytest.raises(
        contract.GeckoLibGenerationContractError,
        match="entrypoints.client must be a list",
    ):
        contract.validate_geckolib_project_preflight(malformed_client)


def test_preflight_rejects_late_build_and_main_source_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(contract, "adapter_from_project", lambda _root: _adapter())

    missing_dependencies = _project_info(
        tmp_path / "build",
        build_text="plugins {}\n",
    )
    with pytest.raises(
        contract.GeckoLibGenerationContractError,
        match="no dependencies block",
    ):
        contract.validate_geckolib_project_preflight(missing_dependencies)

    invalid_main = _project_info(tmp_path / "main")
    invalid_main.main_java.write_bytes(b"\xff\xfe")
    with pytest.raises(
        contract.GeckoLibGenerationContractError,
        match="Fabric main entrypoint must be readable UTF-8",
    ):
        contract.validate_geckolib_project_preflight(invalid_main)


def test_generator_preflight_fails_before_first_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info = _project_info(tmp_path, depends=[])
    writes = {"count": 0}

    monkeypatch.setattr(contract, "adapter_from_project", lambda _root: _adapter())
    monkeypatch.setattr(geckolib_generator, "inspect_fabric_project", lambda _root: info)

    def write_text_files(*_args, **_kwargs):
        writes["count"] += 1
        raise AssertionError("GeckoLib generation must not write after a failed preflight")

    monkeypatch.setattr(geckolib_generator, "write_text_files", write_text_files)
    _install_geckolib_project_preflight(geckolib_generator)

    with pytest.raises(
        geckolib_generator.GeckoLibGenerationError,
        match="generation preflight failed",
    ):
        geckolib_generator.generate_geckolib_entity_assets(
            project_root=tmp_path,
            mod_id="example",
            package_name="com.example",
            entity_id="test_entity",
        )

    assert writes["count"] == 0
