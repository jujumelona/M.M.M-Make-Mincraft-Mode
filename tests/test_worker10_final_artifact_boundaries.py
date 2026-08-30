from __future__ import annotations

import inspect
import zipfile
from pathlib import Path

import pytest

from minecraft_mod_ai import validator
from minecraft_mod_ai.final_artifact import (
    FinalArtifactError,
    _project_root,
    _read_jar_metadata,
    _write_json_receipt,
    append_github_outputs,
    sha256_file,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.validator import validate_jar


def _symlink(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")


def test_sha256_rejects_direct_and_parent_symlink_aliases(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    artifact = real_dir / "artifact.jar"
    artifact.write_bytes(b"artifact")
    assert sha256_file(artifact).startswith("sha256:")

    direct = tmp_path / "direct.jar"
    _symlink(direct, artifact)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        sha256_file(direct)

    alias = tmp_path / "alias"
    _symlink(alias, real_dir, directory=True)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        sha256_file(alias / "artifact.jar")


def test_project_root_rejects_parent_symlink_alias(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    project = real_parent / "project"
    project.mkdir(parents=True)
    alias_parent = tmp_path / "alias-parent"
    _symlink(alias_parent, real_parent, directory=True)

    with pytest.raises(FinalArtifactError, match="symbolic links"):
        _project_root(alias_parent / "project")


def test_receipt_write_rejects_symlink_target_and_parent(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    real_receipt = real_dir / "receipt.json"
    real_receipt.write_text("{}\n", encoding="utf-8")

    direct = tmp_path / "receipt-link.json"
    _symlink(direct, real_receipt)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        _write_json_receipt(direct, {"status": "PASS"})

    alias = tmp_path / "alias"
    _symlink(alias, real_dir, directory=True)
    with pytest.raises(FinalArtifactError, match="unsafe"):
        _write_json_receipt(alias / "new.json", {"status": "PASS"})


def test_github_output_rejects_unsafe_artifact_name(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "mod.jar").write_bytes(b"jar")
    output = tmp_path / "github-output.txt"

    with pytest.raises(FinalArtifactError, match="artifact name"):
        append_github_outputs(
            output,
            {
                "path": str(bundle_dir),
                "artifact": "../mod.jar",
                "artifact_sha256": "sha256:" + "0" * 64,
            },
        )


def test_final_jar_metadata_rejects_windows_drive_entry(tmp_path: Path) -> None:
    jar = tmp_path / "unsafe.jar"
    with zipfile.ZipFile(jar, "w") as archive:
        archive.writestr(
            "fabric.mod.json",
            '{"id":"demo_mod","depends":{"minecraft":"1.21.1"}}',
        )
        archive.writestr("C:/outside.txt", "unsafe")

    with pytest.raises(FinalArtifactError, match="unsafe path"):
        _read_jar_metadata(
            jar,
            loader="fabric",
            metadata_path="fabric.mod.json",
        )


def test_validate_jar_rejects_direct_symlink_before_archive_parsing(tmp_path: Path) -> None:
    spec = MinecraftModPipeline().plan("Create a frost item").spec
    target = tmp_path / "target.jar"
    target.write_bytes(b"not-even-a-zip")
    link = tmp_path / "linked.jar"
    _symlink(link, target)

    report = validate_jar(link, spec)
    assert report.status == "FAIL"
    assert report.findings
    assert report.findings[0].code == "JAR_MISSING"


def test_project_validator_has_one_canonical_boss_validator() -> None:
    source = inspect.getsource(validator.ProjectValidator)
    assert source.count("def _validate_boss(") == 1
