from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.runner import GradleRunner
from minecraft_mod_ai.repairability import source_repair_block_reason


def test_gametest_task_selection_supports_modern_and_legacy_loom(tmp_path: Path) -> None:
    modern = tmp_path / "modern"
    modern.mkdir()
    (modern / "build.gradle").write_text(
        "fabricApi { configureTests() }\n",
        encoding="utf-8",
    )
    assert GradleRunner._configured_gametest_task(modern) == "runGameTest"

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "build.gradle").write_text(
        "loom { runs { gameTestServer { server() } } }\n",
        encoding="utf-8",
    )
    assert GradleRunner._configured_gametest_task(legacy) == "runGameTestServer"

    absent = tmp_path / "absent"
    absent.mkdir()
    (absent / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    assert GradleRunner._configured_gametest_task(absent) is None


def test_missing_gametest_task_is_host_verifier_failure_not_source_repair(tmp_path: Path) -> None:
    log = tmp_path / "gradle-gametest.log"
    log.write_text(
        "org.gradle.execution.TaskSelectionException: "
        "Task 'runGameTest' not found in root project 'fixture'.\n",
        encoding="utf-8",
    )

    assert GradleRunner._gradle_task_missing(log, "runGameTest") is True
    assert (
        source_repair_block_reason(
            build={
                "status": "BLOCKED",
                "error": "Configured GameTest Gradle task is unavailable.",
                "commands": [],
            }
        )
        == "build_status_blocked"
    )
