from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from minecraft_mod_ai import performance_final_contract as contract


def test_snapshot_clone_does_not_use_preflight_rglob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "keep.txt").write_text("keep", encoding="utf-8")
    (project / "nested").mkdir()
    (project / "nested" / "also_keep.json").write_text("{}", encoding="utf-8")
    (project / "build").mkdir()
    (project / "build" / "ignored.txt").write_text("ignore", encoding="utf-8")
    (project / "ignored.png").write_bytes(b"png")

    def forbidden_rglob(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("snapshot staging must not pre-scan with Path.rglob")

    monkeypatch.setattr(Path, "rglob", forbidden_rglob)
    stage = contract._clone_source_snapshot(project)
    try:
        assert (stage / "keep.txt").read_text(encoding="utf-8") == "keep"
        assert (stage / "nested" / "also_keep.json").read_text(encoding="utf-8") == "{}"
        assert not (stage / "build").exists()
        assert not (stage / "ignored.png").exists()
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def test_snapshot_clone_rejects_symlink_in_copy_traversal_and_cleans_stage(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = project / "unsafe-link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(contract.StagedCommitConflict, match="project symlink"):
        contract._clone_source_snapshot(project)

    staging_parent = tmp_path / ".mmm-parallel-staging"
    assert staging_parent.is_dir()
    assert not list(staging_parent.glob("custom-*"))
