from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai.compiler_diagnostics import compiler_log_diagnostics


def test_javac_log_diagnostics_normalize_indented_absolute_source_path(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    source.parent.mkdir(parents=True)
    source.write_text("class DebugToken {}\n", encoding="utf-8")
    log = project / ".minecraft_ai/logs/gradle-build.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        f"  {source}:4: error: package net.minecraft.item does not exist\n"
        f"{source}:7: error: cannot find symbol\n"
        "2 errors\n",
        encoding="utf-8",
    )

    diagnostics = compiler_log_diagnostics(
        {
            "status": "FAIL",
            "commands": [
                {
                    "name": "incremental_build",
                    "exit_code": 1,
                    "timed_out": False,
                    "log_path": str(log),
                }
            ],
        },
        project_root=project,
    )

    assert [item["path"] for item in diagnostics] == [
        "src/main/java/dev/mmm/debugfixture/DebugToken.java",
        "src/main/java/dev/mmm/debugfixture/DebugToken.java",
    ]
    assert [item["line"] for item in diagnostics] == [4, 7]
    assert all(item["source"] == "javac" for item in diagnostics)
    assert diagnostics[0]["message"] == "package net.minecraft.item does not exist"


def test_successful_build_command_yields_no_compiler_diagnostics(tmp_path: Path) -> None:
    log = tmp_path / "gradle-build.log"
    log.write_text("src/main/java/X.java:1: error: ignored\n", encoding="utf-8")

    assert (
        compiler_log_diagnostics(
            {
                "status": "PASS",
                "commands": [
                    {
                        "name": "build",
                        "exit_code": 0,
                        "timed_out": False,
                        "log_path": str(log),
                    }
                ],
            }
        )
        == []
    )


def test_runtime_stack_frame_binds_gametest_failure_to_generated_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src/main/java/dev/mmm/debugfixture/DebugToken.java"
    source.parent.mkdir(parents=True)
    source.write_text("class DebugToken {}\n", encoding="utf-8")
    log = project / ".minecraft_ai/logs/gradle-build.log"
    log.parent.mkdir(parents=True)
    log.write_text(
        "Caused by: java.lang.NullPointerException: Item id not set\n"
        "\tat knot//net.minecraft.world.item.Item.<init>(Item.java:150)\n"
        "\tat knot//dev.mmm.debugfixture.DebugToken.<clinit>(DebugToken.java:13)\n"
        "> Task :runGameTest FAILED\n",
        encoding="utf-8",
    )

    diagnostics = compiler_log_diagnostics(
        {
            "status": "FAIL",
            "commands": [
                {
                    "name": "incremental_build",
                    "exit_code": 1,
                    "timed_out": False,
                    "log_path": str(log),
                }
            ],
        },
        project_root=project,
    )

    runtime = [item for item in diagnostics if item["source"] == "runtime"]
    assert runtime == [
        {
            **runtime[0],
            "path": "src/main/java/dev/mmm/debugfixture/DebugToken.java",
            "line": 13,
            "severity": 1,
            "source": "runtime",
            "message": "java.lang.NullPointerException: Item id not set",
            "code": "runtime:stack:13",
        }
    ]
