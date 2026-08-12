from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.runner_parallel_validation_contract as contract


def test_project_build_lock_is_stable_for_same_path(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    assert contract._path_lock(project) is contract._path_lock(project.resolve())


def test_distribution_marker_is_target_specific(tmp_path: Path) -> None:
    first = contract._marker_path(tmp_path, "8.10.2", "a" * 64)
    second = contract._marker_path(tmp_path, "8.11.1", "b" * 64)
    assert first != second
    assert "8.10.2" in first.name
    assert "8.11.1" in second.name
