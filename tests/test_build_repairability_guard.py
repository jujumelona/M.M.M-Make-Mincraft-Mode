from __future__ import annotations

from pathlib import Path

from minecraft_mod_ai import complete_build_repair
from minecraft_mod_ai import repair_guard
from minecraft_mod_ai.repairability import source_repair_block_reason


class _Report:
    def __init__(self, value):
        self.value = dict(value)

    def to_dict(self):
        return dict(self.value)


def test_unavailable_build_does_not_activate_router_or_repair(monkeypatch, tmp_path: Path) -> None:
    class Runner:
        def __init__(self, _cache):
            pass

        def build(self, _root, *, run_gametest):
            assert run_gametest is True
            return _Report(
                {
                    "status": "UNAVAILABLE",
                    "error": "Java 25 toolchain unavailable for target 26.2",
                    "commands": [],
                }
            )

    class ForbiddenRepairEngine:
        def __init__(self, **_kwargs):
            raise AssertionError("RepairEngine must not be created for UNAVAILABLE builds")

    def forbidden_router_factory():
        raise AssertionError("router must not be activated for UNAVAILABLE builds")

    monkeypatch.setattr(complete_build_repair, "GradleRunner", Runner)
    monkeypatch.setattr(complete_build_repair, "RepairEngine", ForbiddenRepairEngine)

    bundle, router = complete_build_repair.run_build_with_repair(
        project_root=tmp_path,
        cache=tmp_path / ".cache",
        run_gametest=True,
        auto_repair=True,
        max_repair_attempts=None,
        router=None,
        router_factory=forbidden_router_factory,
        policy=object(),
    )

    assert router is None
    assert bundle["build"]["status"] == "UNAVAILABLE"
    assert bundle["repair"]["status"] == "FAIL"
    assert bundle["repair"]["repairable"] is False
    assert bundle["repair"]["stop_reason"] == "non_source_repairable"
    assert bundle["repair"]["reason"] == "build_status_unavailable"


def test_explicit_nonrepairable_failure_does_not_enter_source_repair(
    monkeypatch, tmp_path: Path
) -> None:
    class Runner:
        def __init__(self, _cache):
            pass

        def build(self, _root, *, run_gametest):
            return _Report(
                {
                    "status": "FAIL",
                    "failure_class": "environment",
                    "repairable": False,
                    "error_code": "JAVA_TOOLCHAIN_UNAVAILABLE",
                    "error": "required JDK is absent",
                    "commands": [],
                }
            )

    monkeypatch.setattr(complete_build_repair, "GradleRunner", Runner)

    bundle, router = complete_build_repair.run_build_with_repair(
        project_root=tmp_path,
        cache=tmp_path / ".cache",
        run_gametest=False,
        auto_repair=True,
        max_repair_attempts=None,
        router=None,
        router_factory=lambda: (_ for _ in ()).throw(
            AssertionError("router must not be activated")
        ),
        policy=object(),
    )

    assert router is None
    assert bundle["repair"]["reason"] == "build_marked_non_repairable"


def test_source_failure_still_enters_repair(monkeypatch, tmp_path: Path) -> None:
    calls = {"repair": 0}
    router = object()

    class Runner:
        def __init__(self, _cache):
            pass

        def build(self, _root, *, run_gametest):
            return _Report(
                {
                    "status": "FAIL",
                    "error": "Gradle build failed.",
                    "commands": [
                        {"name": "build", "exit_code": 1, "timed_out": False}
                    ],
                }
            )

    class RepairEngine:
        def __init__(self, **kwargs):
            assert kwargs["router"] is router

        def repair(self, _root, *, run_gametest, max_attempts, initial_build):
            calls["repair"] += 1
            assert initial_build["status"] == "FAIL"
            assert initial_build["error"] == "Gradle build failed."
            return {
                "schema_version": "mmm/repair-result-v2",
                "status": "PASS",
                "attempts": 1,
                "evidence": {
                    "passed": True,
                    "build": {
                        "status": "PASS",
                        "commands": [],
                        "jar_path": "build/libs/example.jar",
                    },
                },
                "patch_receipts": [{"changed": True}],
            }

    monkeypatch.setattr(complete_build_repair, "GradleRunner", Runner)
    monkeypatch.setattr(complete_build_repair, "RepairEngine", RepairEngine)

    bundle, active_router = complete_build_repair.run_build_with_repair(
        project_root=tmp_path,
        cache=tmp_path / ".cache",
        run_gametest=False,
        auto_repair=True,
        max_repair_attempts=None,
        router=None,
        router_factory=lambda: router,
        policy=object(),
    )

    assert calls["repair"] == 1
    assert active_router is router
    assert bundle["build"]["status"] == "PASS"
    assert bundle["repair"]["status"] == "PASS"


def test_classifier_blocks_timeout_unsupported_release_and_validation_mutation() -> None:
    assert (
        source_repair_block_reason(
            build={
                "status": "FAIL",
                "commands": [{"name": "build", "exit_code": 1, "timed_out": True}],
            }
        )
        == "build_command_timed_out"
    )
    assert (
        source_repair_block_reason(
            build={
                "status": "FAIL",
                "error": "error: release version 25 not supported",
                "commands": [],
            }
        )
        == "java_release_not_supported"
    )
    assert (
        source_repair_block_reason(
            build={
                "status": "FAIL",
                "error": "Project inputs changed during validation; result is not certifiable.",
                "commands": [],
            }
        )
        == "validation_inputs_changed"
    )


def test_guarded_repair_engine_never_calls_coder_for_unavailable_jdt(monkeypatch) -> None:
    called = {"base": 0}

    def base_request_patch(self, evidence, context):
        called["base"] += 1
        return [{"operation": "create", "path": "should-not-run"}]

    monkeypatch.setattr(
        repair_guard._BaseRepairEngine,
        "_request_patch",
        base_request_patch,
    )
    engine = object.__new__(repair_guard.RepairEngine)

    result = engine._request_patch(
        {
            "passed": False,
            "build": {
                "status": "FAIL",
                "error": "Gradle build failed.",
                "commands": [
                    {"name": "build", "exit_code": 1, "timed_out": False}
                ],
            },
            "diagnostics": {
                "status": "UNAVAILABLE",
                "error": "TimeoutError: Owner resolve timed out after 90.0s",
                "diagnostics": {},
            },
        },
        {},
    )

    assert result == []
    assert called["base"] == 0


def test_guarded_repair_engine_delegates_for_source_diagnostic(monkeypatch) -> None:
    expected = [{"operation": "replace", "path": "src/main/java/Example.java"}]

    def base_request_patch(self, evidence, context):
        return expected

    monkeypatch.setattr(
        repair_guard._BaseRepairEngine,
        "_request_patch",
        base_request_patch,
    )
    engine = object.__new__(repair_guard.RepairEngine)

    result = engine._request_patch(
        {
            "passed": False,
            "build": {
                "status": "FAIL",
                "error": "Gradle build failed.",
                "commands": [
                    {"name": "build", "exit_code": 1, "timed_out": False}
                ],
            },
            "diagnostics": {
                "status": "FAIL",
                "error": "",
                "diagnostics": {
                    "file:///Example.java": [
                        {
                            "severity": 1,
                            "code": "compiler.err.cant.resolve",
                            "message": "cannot find symbol",
                        }
                    ]
                },
            },
        },
        {},
    )

    assert result == expected
