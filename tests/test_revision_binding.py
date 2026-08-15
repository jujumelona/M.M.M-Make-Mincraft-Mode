from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.importer import inspect_existing_project_archive
from minecraft_mod_ai.spec import SpecValidationError


def _source_zip(
    path: Path,
    *,
    marker: str = "v1",
    minecraft_version: str = "mmm-existing-target",
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "existing/src/main/resources/fabric.mod.json",
            json.dumps(
                {
                    "schemaVersion": 1,
                    "id": "existing_mod",
                    "name": "Existing Mod",
                    "version": "2.3.4",
                    "depends": {"minecraft": minecraft_version},
                }
            ),
        )
        archive.writestr(
            "existing/src/main/java/example/ExistingMod.java",
            f"package example; // {marker}\n",
        )
        archive.writestr("existing/build.gradle", "plugins { id 'fabric-loom' }\n")
    return path


def test_existing_input_snapshot_is_bound_into_the_approval_hash(
    tmp_path: Path,
) -> None:
    archive = _source_zip(tmp_path / "existing.zip")
    pipeline = MinecraftModPipeline()

    fresh = pipeline.plan("Create a frost boss")
    revision = pipeline.plan(
        "Add a frost boss to this project",
        existing_input=archive,
    )

    assert fresh.imported_source_snapshot_hash == ""
    assert revision.imported_source_snapshot_hash.startswith("sha256:")
    assert revision.approval_hash != fresh.approval_hash
    assert any(
        request.capability == "minimal_existing_project_diff"
        for request in revision.deferred_requests
    )


def test_bound_existing_input_is_required_and_rechecked_before_writes(
    tmp_path: Path,
) -> None:
    archive = _source_zip(tmp_path / "existing.zip")
    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan("Add a crafted item", existing_input=archive)
    output_root = tmp_path / "output"

    with pytest.raises(SpecValidationError, match="same ZIP is required"):
        pipeline.execute(
            proposal,
            approval_hash=proposal.approval_hash,
            output_root=output_root,
            build=False,
        )
    assert not output_root.exists()

    _source_zip(archive, marker="changed-after-approval")
    with pytest.raises(SpecValidationError, match="changed after approval"):
        pipeline.execute(
            proposal,
            approval_hash=proposal.approval_hash,
            output_root=output_root,
            build=False,
            existing_input=archive,
        )
    assert not output_root.exists()


def test_existing_project_version_is_preserved_without_silent_migration(
    tmp_path: Path,
) -> None:
    archive = _source_zip(
        tmp_path / "future.zip",
        minecraft_version="mmm-future-target",
    )

    proposal = MinecraftModPipeline().plan(
        "Add a crafted item",
        existing_input=archive,
    )

    assert proposal.spec.platform.minecraft_version == "mmm-future-target"
    assert proposal.spec.platform.loader == "fabric"
    assert proposal.imported_source_snapshot_hash.startswith("sha256:")


def test_revision_candidate_release_contains_import_inventory(
    tmp_path: Path,
) -> None:
    archive = _source_zip(tmp_path / "existing.zip")
    pipeline = MinecraftModPipeline()
    proposal = pipeline.plan(
        "Add a frost item and block",
        existing_input=archive,
    )

    result = pipeline.execute(
        proposal,
        approval_hash=proposal.approval_hash,
        output_root=tmp_path / "output",
        build=False,
        existing_input=archive,
    )

    report_path = (
        Path(result.release_dir)
        / "evidence"
        / "imported-project-inventory.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.existing_input_kind == "source_project"
    assert (
        result.imported_source_snapshot_hash
        == proposal.imported_source_snapshot_hash
        == report["source_snapshot_hash"]
    )
    assert report["mod_id"] == "existing_mod"
    assert report["trusted_generated_source"] is False


def test_generated_release_bundle_recovers_nested_editable_source_inventory(
    tmp_path: Path,
) -> None:
    metadata = {
        "schemaVersion": 1,
        "id": "nested_existing",
        "name": "Nested Existing",
        "version": "4.5.6",
        "depends": {"minecraft": "mmm-nested-target"},
    }
    source_bytes = io.BytesIO()
    with zipfile.ZipFile(source_bytes, "w") as source:
        source.writestr(
            "src/main/resources/fabric.mod.json",
            json.dumps(metadata),
        )
        source.writestr(
            "src/main/java/example/NestedExisting.java",
            "package example;\n",
        )
        source.writestr("build.gradle", "plugins { id 'fabric-loom' }\n")

    jar_bytes = io.BytesIO()
    with zipfile.ZipFile(jar_bytes, "w") as jar:
        jar.writestr("fabric.mod.json", json.dumps(metadata))
        jar.writestr("example/NestedExisting.class", b"\xca\xfe\xba\xbe")

    release = tmp_path / "nested-release.zip"
    with zipfile.ZipFile(release, "w") as archive:
        archive.writestr(
            "source/nested_existing-4.5.6-source.zip",
            source_bytes.getvalue(),
        )
        archive.writestr(
            "binaries/nested_existing-4.5.6.jar",
            jar_bytes.getvalue(),
        )

    report = inspect_existing_project_archive(release)

    assert report.input_kind == "source_and_release"
    assert report.has_sources is True
    assert report.jar_only is False
    assert report.mod_id == "nested_existing"
    assert any("source.zip!/src/main/java/" in path for path in report.source_files)
    assert any("source.zip!/build.gradle" in path for path in report.gradle_files)
