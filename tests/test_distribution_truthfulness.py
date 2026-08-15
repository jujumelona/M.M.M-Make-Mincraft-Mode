from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai.generator import FabricProjectGenerator
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.publisher import (
    PublishingError,
    build_distribution_metadata,
    dependency_inventory_from_metadata,
    package_distribution_bundle,
    publish_modrinth,
)
from minecraft_mod_ai.runner import BuildReport
from minecraft_mod_ai.toolchain_contract import fabric_dependency_predicates
from minecraft_mod_ai.validator import ValidationReport


def _jar(path: Path, metadata: dict[str, object]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "fabric.mod.json",
            json.dumps(metadata, separators=(",", ":")),
        )
        archive.writestr("example/Entry.class", b"\xca\xfe\xba\xbe")
    return path


def _platform_lock():
    return MinecraftModPipeline().plan("Create one truth item").spec.platform


def _metadata(
    *,
    mod_id: str = "truth_test",
    version: str = "1.0.0",
) -> dict[str, object]:
    platform = _platform_lock()
    return {
        "schemaVersion": 1,
        "id": mod_id,
        "version": version,
        "environment": "*",
        "depends": fabric_dependency_predicates(platform),
    }


def test_generator_declares_only_the_tested_platform_predicates(
    tmp_path: Path,
) -> None:
    spec = MinecraftModPipeline().plan("Create one truth item").spec
    FabricProjectGenerator().generate(spec, tmp_path)

    metadata = json.loads(
        (tmp_path / "src/main/resources/fabric.mod.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["depends"] == fabric_dependency_predicates(spec.platform)
    assert all(
        not value.startswith((">", "<", "~", "^"))
        for value in metadata["depends"].values()
    )


def test_distribution_preserves_geckolib_custom_and_modrinth_dependencies(
    tmp_path: Path,
) -> None:
    fabric = _metadata()
    fabric["depends"]["geckolib"] = "4.8.2"  # type: ignore[index]
    fabric["recommends"] = {"custom-library": "^2.4.0"}
    jar = _jar(tmp_path / "truth.jar", fabric)

    metadata = build_distribution_metadata(
        jar_path=jar,
        mod_id="truth_test",
        version="1.0.0",
        name="Truth Test",
        changelog="Verified dependency metadata.",
        platform_lock=_platform_lock(),
        modrinth_project_ids={"custom-library": "AbC12345"},
    )

    by_id = {item["mod_id"]: item for item in metadata["fabric_dependencies"]}
    assert by_id["fabric-api"]["version_predicates"] == ["test-api"]
    assert by_id["geckolib"]["version_predicates"] == ["4.8.2"]
    assert by_id["custom-library"]["fabric_section"] == "recommends"
    assert metadata["modrinth_dependencies"] == [
        {"project_id": "8BmcQJ2H", "dependency_type": "required"},
        {"project_id": "AbC12345", "dependency_type": "optional"},
        {"project_id": "P7dR8mSH", "dependency_type": "required"},
    ]

    bundle = tmp_path / "distribution.zip"
    package_distribution_bundle(metadata, output_zip=bundle)
    with zipfile.ZipFile(bundle) as archive:
        sbom = json.loads(archive.read("supply_chain/sbom.cdx.json"))
        provenance = json.loads(archive.read("supply_chain/provenance.json"))
    assert {component["name"] for component in sbom["components"]} >= {
        "Fabric API",
        "GeckoLib",
        "custom-library",
    }
    assert {
        item["mod_id"] for item in provenance["declared_fabric_dependencies"]
    } >= {"fabric-api", "geckolib", "custom-library"}


def test_unknown_custom_dependency_blocks_modrinth_instead_of_being_omitted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _metadata()
    fabric["depends"]["private-runtime"] = "1.0.0"  # type: ignore[index]
    metadata = build_distribution_metadata(
        jar_path=_jar(tmp_path / "custom.jar", fabric),
        mod_id="truth_test",
        version="1.0.0",
        name="Truth Test",
        changelog="Custom runtime.",
        platform_lock=_platform_lock(),
    )
    monkeypatch.setenv("MODRINTH_TOKEN", "not-used")

    with pytest.raises(PublishingError, match="private-runtime"):
        publish_modrinth(metadata, project_id="Project1")


def test_modrinth_upload_uses_resolved_declared_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric = _metadata()
    fabric["depends"]["geckolib"] = "4.8.2"  # type: ignore[index]
    metadata = build_distribution_metadata(
        jar_path=_jar(tmp_path / "resolved.jar", fabric),
        mod_id="truth_test",
        version="1.0.0",
        name="Truth Test",
        changelog="Resolved dependencies.",
        platform_lock=_platform_lock(),
    )
    captured: dict[str, object] = {}

    class Response:
        status_code = 201
        text = ""

        @staticmethod
        def json() -> dict[str, str]:
            return {"id": "Version1"}

    class Client:
        def __init__(self, *, timeout: int) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def post(self, _url, *, headers, files):
            captured["headers"] = headers
            captured["data"] = json.loads(files["data"][1])
            return Response()

    monkeypatch.setenv("MODRINTH_TOKEN", "test-token")
    monkeypatch.setattr("minecraft_mod_ai.publisher.httpx.Client", Client)

    receipt = publish_modrinth(metadata, project_id="Project1")

    assert receipt["status"] == "PUBLISHED"
    assert captured["data"]["dependencies"] == [  # type: ignore[index]
        {"project_id": "8BmcQJ2H", "dependency_type": "required"},
        {"project_id": "P7dR8mSH", "dependency_type": "required"},
    ]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.update({"depends": []}), "depends must be an object"),
        (
            lambda value: value["depends"].update({"fabric-api": ">=0.92.11"}),
            "tested predicate",
        ),
        (
            lambda value: value["depends"].update({"broken-library": {"v": "1"}}),
            "string or non-empty string list",
        ),
    ],
)
def test_invalid_or_unbounded_dependency_metadata_fails_closed(
    mutation,
    message: str,
) -> None:
    fabric = _metadata()
    mutation(fabric)
    platform = _platform_lock()

    with pytest.raises(PublishingError, match=message):
        dependency_inventory_from_metadata(fabric, platform_lock=platform)


def test_release_sbom_and_provenance_use_declared_source_metadata(
    tmp_path: Path,
) -> None:
    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan("Create one truth item")
    project = tmp_path / "project"
    FabricProjectGenerator().generate(proposal.spec, project)
    metadata_path = project / "src/main/resources/fabric.mod.json"
    fabric = json.loads(metadata_path.read_text(encoding="utf-8"))
    fabric["depends"]["geckolib"] = "4.8.2"
    fabric["suggests"] = {"custom-library": "^2.4.0"}
    metadata_path.write_text(json.dumps(fabric), encoding="utf-8")

    release_dir, _, _ = pipeline._package_release(
        proposal,
        project,
        tmp_path / "releases",
        ValidationReport(status="PASS", checks_run=1, findings=()),
        BuildReport(
            status="NOT_RUN",
            gradle_version=proposal.spec.platform.gradle,
            commands=(),
            jar_path=None,
            gametest_report=None,
        ),
        ValidationReport(status="NOT_RUN", checks_run=0, findings=()),
        validated_jar_sha256=None,
        existing_report=None,
    )

    sbom = json.loads(
        (release_dir / "supply_chain/sbom.cdx.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (release_dir / "supply_chain/provenance.json").read_text(encoding="utf-8")
    )
    components = {component["name"]: component for component in sbom["components"]}
    assert components["GeckoLib"]["version"] == "4.8.2"
    assert components["custom-library"]["scope"] == "optional"
    assert {
        item["mod_id"] for item in provenance["declared_fabric_dependencies"]
    } >= {"fabric-api", "geckolib", "custom-library"}
