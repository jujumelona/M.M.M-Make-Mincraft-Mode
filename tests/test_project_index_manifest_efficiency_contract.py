from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from minecraft_mod_ai.project_index import ProjectIndex


project_index_module = importlib.import_module("minecraft_mod_ai.project_index")


def _indexed_project(root: Path, count: int = 300) -> ProjectIndex:
    source = root / "src/main/java/example"
    source.mkdir(parents=True)
    for index in range(count):
        (source / f"Generated{index:04d}.java").write_text(
            f"package example; final class Generated{index:04d} {{}}\n",
            encoding="utf-8",
        )
    return ProjectIndex(root)


def _manifest(index: ProjectIndex) -> tuple[Path, dict]:
    path = index.write_manifest()
    return path, json.loads(path.read_text(encoding="utf-8"))


def _part_path(manifest_path: Path, record: dict) -> Path:
    return manifest_path.parent / record["path"]


def test_incremental_path_updates_preserve_sorted_tuple_and_exact_membership(
    tmp_path: Path,
) -> None:
    index = _indexed_project(tmp_path, count=16)
    assert getattr(ProjectIndex.update_files, "_mmm_incremental_sorted_update", False)

    modified = tmp_path / "src/main/java/example/Generated0008.java"
    modified.write_text(
        "package example; final class Generated0008 { int value = 8; }\n",
        encoding="utf-8",
    )
    inserted = tmp_path / "src/main/java/example/Generated0008A.java"
    inserted.write_text(
        "package example; final class Generated0008A {}\n",
        encoding="utf-8",
    )
    removed = tmp_path / "src/main/java/example/Generated0003.java"
    removed.unlink()

    index.update_files((modified, inserted, removed))

    assert isinstance(index.files, tuple)
    paths = [item.path for item in index.files]
    assert paths == sorted(paths)
    assert "src/main/java/example/Generated0008A.java" in paths
    assert "src/main/java/example/Generated0003.java" not in paths
    assert index._by_path["src/main/java/example/Generated0008.java"].sha256 == (
        "sha256:" + hashlib.sha256(modified.read_bytes()).hexdigest()
    )


def test_same_process_exact_snapshot_is_zero_write_fast_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = _indexed_project(tmp_path)
    manifest_path, first = _manifest(index)

    def forbidden_replace(*_args, **_kwargs):
        raise AssertionError("exact committed ProjectIndex receipt must not rewrite disk")

    monkeypatch.setattr(project_index_module.os, "replace", forbidden_replace)
    assert index.write_manifest() == manifest_path
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == first


def test_changed_snapshot_reuses_verified_unchanged_shards(
    tmp_path: Path,
) -> None:
    index = _indexed_project(tmp_path)
    first_path, first = _manifest(index)
    assert len(first["parts"]) == 2
    old_part0 = _part_path(first_path, first["parts"][0])
    old_inode = old_part0.stat().st_ino

    changed = tmp_path / "src/main/java/example/Generated0299.java"
    changed.write_text(
        "package example; final class Generated0299 { int changed = 1; }\n",
        encoding="utf-8",
    )
    index.update_files((changed,))
    second_path, second = _manifest(index)
    assert second["sha256"] != first["sha256"]
    assert len(second["parts"]) == 2

    new_part0 = _part_path(second_path, second["parts"][0])
    if os.name == "posix":
        assert new_part0.stat().st_ino == old_inode

    for record in second["parts"]:
        part = _part_path(second_path, record)
        assert record["sha256"] == (
            "sha256:" + hashlib.sha256(part.read_bytes()).hexdigest()
        )


def test_resumed_process_rejects_corrupt_snapshot_before_fast_path(
    tmp_path: Path,
) -> None:
    index = _indexed_project(tmp_path)
    manifest_path, manifest = _manifest(index)
    part = _part_path(manifest_path, manifest["parts"][0])
    part.write_text("corrupt\n", encoding="utf-8")

    resumed = ProjectIndex(tmp_path)
    with pytest.raises(OSError, match="digest mismatch"):
        resumed.write_manifest()
