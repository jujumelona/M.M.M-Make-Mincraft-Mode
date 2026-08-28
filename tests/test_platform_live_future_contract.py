from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai import platform_validation_contract as live_validation
from minecraft_mod_ai.fabric_official_template_provider import (
    _gradle_wrapper_version,
    _java_release,
)
from minecraft_mod_ai.platform_catalog import PlatformAdapter
from minecraft_mod_ai.validator import ProjectValidator


def _live_adapter() -> PlatformAdapter:
    return PlatformAdapter(
        adapter_id="fabric_live_27_0_test",
        edition="java",
        loader="fabric",
        minecraft_version="27.0",
        java_version="25",
        yarn_mappings="mojang",
        mappings_kind="mojang",
        mappings_version="mojang",
        fabric_loader="0.20.0",
        fabric_api="0.200.0+27.0",
        fabric_loom="1.20-SNAPSHOT",
        gradle="9.7",
        gradle_sha256="a" * 64,
        data_pack_version="100.0",
        resource_pack_version="100.0",
        resource_pack_format=0,
        release_metadata_url="https://www.minecraft.net/en-us/article/minecraft-java-edition-27-0",
        source_api_family="fabric_live_ai",
        deterministic_module_kinds=frozenset(),
    )


def _write_live_project(root: Path) -> None:
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True)
    (root / "gradle/wrapper").mkdir(parents=True)
    (root / "src/main/java/example/ExampleMod.java").write_text(
        "package example; public final class ExampleMod {}\n",
        encoding="utf-8",
    )
    (root / "gradle.properties").write_text(
        "\n".join(
            (
                "minecraft_version=27.0",
                "loader_version=0.20.0",
                "loom_version=1.20-SNAPSHOT",
                "fabric_api_version=0.200.0+27.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "src/main/resources/fabric.mod.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "future_probe",
                "version": "${version}",
                "environment": "*",
                "entrypoints": {"main": ["example.ExampleMod"]},
                "depends": {
                    "fabricloader": ">=0.20.0",
                    "minecraft": "~27.0",
                    "java": ">=25",
                    "fabric-api": "*",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "gradle/wrapper/gradle-wrapper.properties").write_text(
        "distributionUrl=https\\://services.gradle.org/distributions/gradle-9.7-bin.zip\n",
        encoding="utf-8",
    )
    (root / "build.gradle").write_text(
        "tasks.withType(JavaCompile).configureEach { it.options.release = 25 }\n",
        encoding="utf-8",
    )


def test_live_validator_does_not_guess_future_pack_format_or_legacy_paths(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "future"
    _write_live_project(root)
    adapter = _live_adapter()
    monkeypatch.setattr(live_validation, "adapter_for_lock_values", lambda _value: adapter)
    monkeypatch.setattr(live_validation, "adapter_from_project", lambda _root: adapter)
    spec = SimpleNamespace(
        platform=SimpleNamespace(),
        mod_id="future_probe",
    )

    report = ProjectValidator().validate(root, spec)
    assert report.status == "PASS", [item.__dict__ for item in report.findings]
    assert not any(item.code == "BAD_RESOURCE_PACK_FORMAT" for item in report.findings)


def test_official_bootstrap_toolchain_parsers_bind_gradle_and_java(tmp_path: Path) -> None:
    root = tmp_path / "future"
    _write_live_project(root)
    assert _gradle_wrapper_version(root) == "9.7"
    assert _java_release(root) == "25"
