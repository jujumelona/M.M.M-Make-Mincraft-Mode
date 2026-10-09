"""Regression tests for verified end-user download bundle symlink handling."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.colab_run_modes import _user_download_zip


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _result(root: Path, *, resource: bool = False) -> SimpleNamespace:
    artifact_sha = _sha256((root / "generated-mod.jar").read_bytes())
    members = [{"path": "generated-mod.jar", "sha256": artifact_sha}]
    additional = {}
    if resource:
        pack_sha = _sha256((root / "generated-resource-pack.zip").read_bytes())
        members.append({"path": "generated-resource-pack.zip", "sha256": pack_sha})
        additional["generated-resource-pack.zip"] = pack_sha
    return SimpleNamespace(
        complete_proposal_hash="verified-proposal",
        distribution_receipt={
            "downloadable_bundle": {
                "status": "PASS",
                "path": str(root),
                "artifact": "generated-mod.jar",
                "artifact_sha256": artifact_sha,
                "proposal_hash": "verified-proposal",
                "members": members,
                "additional_artifacts": additional,
            }
        },
    )


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable on this runner: {exc}")


def test_verified_regular_bundle_creates_only_expected_download(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "generated-mod.jar").write_bytes(b"verified mod jar")

    archive_path = _user_download_zip(_result(root))
    assert archive_path is not None
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == ["generated-mod.jar"]
        assert archive.read("generated-mod.jar") == b"verified mod jar"


def test_reject_symlinked_primary_artifact_even_with_matching_hash(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    actual = root / "real-mod.jar"
    actual.write_bytes(b"same verified bytes")
    _symlink(root / "generated-mod.jar", actual)

    assert _user_download_zip(_result(root)) is None
    assert not (tmp_path / "bundle-user.zip").exists()


def test_reject_symlinked_bundle_directory_even_with_matching_hash(tmp_path: Path) -> None:
    root = tmp_path / "real-bundle"
    root.mkdir()
    (root / "generated-mod.jar").write_bytes(b"verified mod jar")
    alias = tmp_path / "bundle-link"
    _symlink(alias, root)

    assert _user_download_zip(_result(alias)) is None
    assert not (tmp_path / "real-bundle-user.zip").exists()


def test_reject_symlinked_optional_resource_pack(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "generated-mod.jar").write_bytes(b"verified mod jar")
    actual_pack = root / "original-pack.zip"
    actual_pack.write_bytes(b"verified pack")
    _symlink(root / "generated-resource-pack.zip", actual_pack)

    assert _user_download_zip(_result(root, resource=True)) is None
    assert not (tmp_path / "bundle-user.zip").exists()
