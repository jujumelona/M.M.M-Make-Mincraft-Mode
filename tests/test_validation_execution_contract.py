from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

from minecraft_mod_ai import extended_content_generator
from minecraft_mod_ai.extended_registration_contract import _replace_registration_method
from minecraft_mod_ai.java_lsp import JavaLanguageService
from minecraft_mod_ai.repair_engine import RepairEngine
from minecraft_mod_ai.runner import BuildReport, GradleRunner
from minecraft_mod_ai.validation_execution_contract import (
    _diagnostic_errors,
    gametest_resource_errors,
    project_build_fingerprint,
)


def _project(root: Path, mod_id: str = "example_mod") -> Path:
    (root / "src/main/java/example").mkdir(parents=True)
    (root / "src/main/resources").mkdir(parents=True)
    (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
    (root / "settings.gradle").write_text("rootProject.name='example'\n", encoding="utf-8")
    (root / "gradle.properties").write_text("org.gradle.jvmargs=-Xmx1g\n", encoding="utf-8")
    (root / "src/main/java/example/Main.java").write_text(
        "package example; public final class Main {}\n",
        encoding="utf-8",
    )
    (root / "src/main/resources/fabric.mod.json").write_text(
        '{"schemaVersion":1,"id":"' + mod_id + '","version":"1.0.0"}\n',
        encoding="utf-8",
    )
    return root


def test_validation_runtime_contracts_are_installed() -> None:
    assert getattr(GradleRunner.build, "_mmm_incremental_validation", False)
    assert getattr(GradleRunner.build, "_mmm_exact_input_cache", False)
    assert getattr(GradleRunner._ensure_gradle, "_mmm_target_parallel_distribution", False)
    assert getattr(JavaLanguageService.diagnostics, "_mmm_exact_java_cache", False)
    assert getattr(RepairEngine._evidence, "_mmm_progressive_evidence", False)
    assert getattr(RepairEngine._request_patch, "_mmm_tracks_repair_scope", False)
    assert getattr(
        extended_content_generator.generate_extended_content,
        "_mmm_static_registrar_tree",
        False,
    )


def test_build_fingerprint_ignores_outputs_and_logs_but_not_sources(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    first = project_build_fingerprint(root)

    (root / "build/classes").mkdir(parents=True)
    (root / "build/classes/output.bin").write_bytes(b"ignored")
    (root / ".minecraft_ai/logs").mkdir(parents=True)
    (root / ".minecraft_ai/logs/gradle.log").write_text("ignored\n", encoding="utf-8")
    assert project_build_fingerprint(root) == first

    (root / "src/main/java/example/Main.java").write_text(
        "package example; public final class Main { int value = 1; }\n",
        encoding="utf-8",
    )
    assert project_build_fingerprint(root) != first


def test_exact_input_successful_build_is_reused_in_process(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")
    runner = GradleRunner(tmp_path / "cache")
    calls = {"count": 0}

    def fake_build_locked(self, project_root: Path, *, run_gametest: bool):
        calls["count"] += 1
        return BuildReport(
            status="PASS",
            gradle_version="test",
            commands=(),
            jar_path=str(project_root / "build/libs/example.jar"),
            gametest_report=None,
            error=None,
        )

    runner._build_locked = MethodType(fake_build_locked, runner)
    first = runner.build(root, run_gametest=False)
    second = runner.build(root, run_gametest=False)
    assert first.status == second.status == "PASS"
    assert calls["count"] == 1

    (root / "src/main/java/example/Main.java").write_text(
        "package example; public final class Main { int changed = 1; }\n",
        encoding="utf-8",
    )
    runner.build(root, run_gametest=False)
    assert calls["count"] == 2


def test_gametest_resource_gate_detects_generated_namespace_errors(tmp_path: Path) -> None:
    root = _project(tmp_path / "project", mod_id="frost_works")
    log = tmp_path / "gametest.log"
    log.write_text(
        "\n".join(
            [
                "[main/ERROR] (Minecraft) Failed to load properties from file: server.properties",
                "[Worker-Main-2/ERROR] (Minecraft) Couldn't parse element loot_tables:frost_works:blocks/reference_machine",
                "com.google.gson.JsonSyntaxException: Expected name to be an item, was unknown string 'frost_works:reference_machine'",
                "[main/ERROR] (Minecraft) Parsing error loading recipe frost_works:frost_crystal",
                "[main/ERROR] (Minecraft) Parsing error loading recipe other_mod:not_ours",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    findings = gametest_resource_errors(root, log)
    assert len(findings) == 2
    assert all("frost_works:" in finding for finding in findings)
    assert not any("server.properties" in finding for finding in findings)
    assert not any("other_mod:" in finding for finding in findings)


def test_jdt_mapping_errors_are_flattened_without_blocking_warnings() -> None:
    receipt = {
        "diagnostics": {
            "file:///a.java": [
                {"severity": 1, "message": "compile error"},
                {"severity": 3, "message": "info"},
            ],
            "file:///b.java": [{"severity": 2, "message": "warning"}],
        }
    }
    errors = _diagnostic_errors(receipt)
    assert [item["message"] for item in errors] == ["compile error"]


def test_progressive_repair_skips_gradle_when_jdt_is_not_clean(tmp_path: Path) -> None:
    root = _project(tmp_path / "project")

    class Diagnostics:
        def diagnostics(self, *_args, **_kwargs):
            return {
                "diagnostics": {
                    "file:///Main.java": [
                        {"severity": 1, "message": "cannot resolve symbol"}
                    ]
                }
            }

    class Runner:
        def __init__(self, _cache):
            raise AssertionError("Gradle must not start while JDT errors remain")

    repair = RepairEngine(
        router=SimpleNamespace(),
        gradle_cache=tmp_path / "cache",
        diagnostics_factory=Diagnostics,
        runner_factory=Runner,
    )
    evidence = repair._evidence(root, run_gametest=True)
    assert evidence["passed"] is False
    assert evidence["build"]["status"] == "SKIPPED"


def test_static_registrar_replaces_runtime_reflection_with_bounded_root() -> None:
    source = '''public final class GeneratedExtendedContent {\n    @SuppressWarnings("unchecked")\n    private static List<Block> registerGeneratedUnits() {\n        Set<String> classes = new TreeSet<>();\n        return new ArrayList<>();\n    }\n\n    public record MachineDefinition(Identifier input) {}\n}\n'''
    updated = _replace_registration_method(source, "GeneratedContentDispatchL000N00000000")
    assert "Class.forName" not in updated
    assert "TreeSet" not in updated.split("private static List<Block> registerGeneratedUnits()", 1)[1].split("public record", 1)[0]
    assert "GeneratedContentDispatchL000N00000000.register();" in updated
    assert "machineBlocks.addAll(GeneratedContentDispatchL000N00000000.machineBlocks());" in updated
