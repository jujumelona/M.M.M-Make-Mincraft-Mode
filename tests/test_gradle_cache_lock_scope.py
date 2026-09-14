from __future__ import annotations

from pathlib import Path

import minecraft_mod_ai.runner as runner_module
from minecraft_mod_ai.runner import GradleRunner


def test_gradle_build_does_not_hold_mmm_distribution_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = GradleRunner(tmp_path / "cache")
    sentinel = object()
    calls: list[tuple[Path, bool]] = []

    def fake_build_locked(project_root: Path, *, run_gametest: bool):
        calls.append((project_root, run_gametest))
        return sentinel

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("whole-build MMM cache lock must not be acquired")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(runner, "_build_locked", fake_build_locked)
    monkeypatch.setattr(
        runner_module,
        "_exclusive_cache_lock",
        lambda *_args, **_kwargs: ForbiddenLock(),
    )

    project = tmp_path / "project"
    assert runner.build(project, run_gametest=False) is sentinel
    assert calls == [(project, False)]
    assert getattr(GradleRunner.build, "_mmm_narrow_gradle_cache_lock", False)
    assert getattr(GradleRunner._ensure_gradle, "_mmm_distribution_lock_owner", False)
