from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

import minecraft_mod_ai.java_diagnostics_fallback_contract as contract


@dataclass(frozen=True)
class FakeCommand:
    name: str = "build"
    command: tuple[str, ...] = ("gradle", "build")
    exit_code: int = 0
    duration_seconds: float = 0.1
    log_path: str = ""
    timed_out: bool = False


class FakeReport:
    def __init__(
        self,
        *,
        status: str,
        commands: tuple[FakeCommand, ...] = (),
        error: str | None = None,
    ) -> None:
        self.status = status
        self.commands = commands
        self.error = error

    @property
    def passed(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "commands": [
                {
                    "name": item.name,
                    "command": list(item.command),
                    "exit_code": item.exit_code,
                    "duration_seconds": item.duration_seconds,
                    "log_path": item.log_path,
                    "timed_out": item.timed_out,
                }
                for item in self.commands
            ],
            "error": self.error,
        }


def unavailable_receipt() -> dict[str, object]:
    return {
        "status": "UNAVAILABLE",
        "error": "JDT diagnostics unavailable",
        "diagnostics": {},
    }


def service_type(java_diagnostics):
    return type(
        "FakeService",
        (),
        {
            "java_diagnostics": java_diagnostics,
            "_existing_dir": lambda self, root: Path(root),
            "workspace_root": "/tmp/mmm-workspace",
        },
    )


def install_runner(monkeypatch, reports):
    state = {"timeouts": [], "builds": 0}
    queue = list(reports)

    class FakeRunner:
        def __init__(self, cache_dir, *, command_timeout_seconds):
            state["timeouts"].append(command_timeout_seconds)

        def build(self, project_root, *, run_gametest):
            state["builds"] += 1
            return queue.pop(0)

    monkeypatch.setattr(contract, "GradleRunner", FakeRunner)
    monkeypatch.setattr(contract, "emit_root_cause", lambda *args, **kwargs: None)
    return state


def test_unavailable_receipt_falls_back_to_gradle_pass(monkeypatch, tmp_path):
    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        return unavailable_receipt()

    service = service_type(jdt)
    state = install_runner(monkeypatch, [FakeReport(status="PASS")])
    contract.install(service)

    result = service().java_diagnostics(str(tmp_path), timeout_seconds=1)

    assert result["status"] == "PASS"
    assert result["verifier"] == "gradle_build_fallback"
    assert state["builds"] == 1
    assert state["timeouts"] == [contract._GRADLE_FALLBACK_TIMEOUT_SECONDS]
    assert state["timeouts"][0] != 1


def test_recognized_jdt_exception_falls_back(monkeypatch, tmp_path):
    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        raise RuntimeError(
            "TOOL_RUNTIME_UNAVAILABLE: JDT LS did not publish diagnostics "
            "before the validation deadline"
        )

    service = service_type(jdt)
    state = install_runner(monkeypatch, [FakeReport(status="PASS")])
    contract.install(service)

    result = service().java_diagnostics(str(tmp_path))

    assert result["status"] == "PASS"
    assert "RuntimeError" in result["jdt_unavailable_reason"]
    assert state["builds"] == 1


def test_unrelated_jdt_exception_is_not_swallowed(monkeypatch, tmp_path):
    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        raise ValueError("bad caller")

    service = service_type(jdt)
    state = install_runner(monkeypatch, [FakeReport(status="PASS")])
    contract.install(service)

    with pytest.raises(ValueError, match="bad caller"):
        service().java_diagnostics(str(tmp_path))

    assert state["builds"] == 0


def test_gradle_compile_failure_is_fail_and_contains_compiler_log(
    monkeypatch, tmp_path
):
    log = tmp_path / "gradle-build.log"
    log.write_text(
        "DebugToken.java:17: error: cannot find symbol\n"
        "    missingMethod();\n"
        "    ^\n",
        encoding="utf-8",
    )

    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        return unavailable_receipt()

    report = FakeReport(
        status="FAIL",
        commands=(FakeCommand(exit_code=1, log_path=str(log)),),
        error="Gradle build failed.",
    )
    service = service_type(jdt)
    install_runner(monkeypatch, [report])
    contract.install(service)

    result = service().java_diagnostics(str(tmp_path))

    assert result["status"] == "FAIL"
    message = result["diagnostics"][0]["message"]
    assert "Gradle build failed." in message
    assert "DebugToken.java:17: error: cannot find symbol" in message
    assert result["diagnostics"][0]["code"] == "GRADLE_BUILD_FAILED"


def test_gradle_timeout_is_infrastructure_timeout(monkeypatch, tmp_path):
    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        return unavailable_receipt()

    report = FakeReport(
        status="FAIL",
        commands=(FakeCommand(exit_code=-9, timed_out=True),),
        error="Gradle build timed out.",
    )
    service = service_type(jdt)
    install_runner(monkeypatch, [report])
    contract.install(service)

    result = service().java_diagnostics(str(tmp_path))

    assert result["status"] == "TIMEOUT"
    assert result["diagnostics"][0]["code"] == "GRADLE_BUILD_TIMEOUT"


def test_install_is_idempotent(monkeypatch, tmp_path):
    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        return unavailable_receipt()

    service = service_type(jdt)
    state = install_runner(monkeypatch, [FakeReport(status="PASS")])
    contract.install(service)
    first = service.java_diagnostics
    contract.install(service)

    assert service.java_diagnostics is first
    assert service().java_diagnostics(str(tmp_path))["status"] == "PASS"
    assert state["builds"] == 1


def test_revalidation_can_fail_then_pass_through_same_fallback(
    monkeypatch, tmp_path
):
    def jdt(self, project_root, relative_files=None, timeout_seconds=60):
        return unavailable_receipt()

    service = service_type(jdt)
    state = install_runner(
        monkeypatch,
        [
            FakeReport(
                status="FAIL",
                commands=(FakeCommand(exit_code=1),),
                error="Gradle build failed.",
            ),
            FakeReport(status="PASS"),
        ],
    )
    contract.install(service)
    instance = service()

    before_repair = instance.java_diagnostics(str(tmp_path))
    after_repair = instance.java_diagnostics(str(tmp_path))

    assert before_repair["status"] == "FAIL"
    assert after_repair["status"] == "PASS"
    assert state["builds"] == 2
