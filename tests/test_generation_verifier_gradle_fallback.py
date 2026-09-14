from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai import generation_verifier_resilience
from minecraft_mod_ai.generation_verifier_fallback_installation import _gradle_fallback_receipt


class _FakeReport:
    def __init__(self, *, passed: bool, log_path: str = "") -> None:
        self.passed = passed
        self.status = "PASS" if passed else "FAIL"
        self.gradle_version = "8.14"
        self.error = None if passed else "Gradle build failed."
        self._log_path = log_path

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "gradle_version": self.gradle_version,
            "commands": ([{"name": "build", "log_path": self._log_path}] if self._log_path else []),
            "jar_path": "build/libs/example.jar" if self.passed else None,
            "gametest_report": None,
            "error": self.error,
        }


class _FakeRunner:
    report: _FakeReport

    def __init__(self, _cache_dir) -> None:
        pass

    def build(self, _project_root, *, run_gametest: bool = True) -> _FakeReport:
        assert run_gametest is False
        return self.report


def _runtime_module():
    return SimpleNamespace(_bounded_result=lambda value: value)


def test_finalized_generation_verifier_has_gradle_fallback_installed() -> None:
    assert getattr(
        generation_verifier_resilience.run_generation_verifier,
        "_mmm_generation_gradle_fallback",
        False,
    ) is True


def test_gradle_fallback_pass_is_real_verifier_receipt(tmp_path) -> None:
    _FakeRunner.report = _FakeReport(passed=True)
    runtime = SimpleNamespace(workspace_root=str(tmp_path))

    receipt = _gradle_fallback_receipt(
        runtime,
        tmp_path,
        runtime_module=_runtime_module(),
        jdt_error=RuntimeError("owner unavailable"),
        gradle_runner_factory=_FakeRunner,
    )

    assert receipt["status"] == "PASS"
    assert receipt["verifier_backend"] == "gradle_build"
    assert receipt["fallback_from"] == "java_diagnostics"
    assert receipt["diagnostics"] == []
    assert receipt["error_count"] == 0


def test_gradle_fallback_failure_returns_actionable_diagnostic(tmp_path) -> None:
    log_path = tmp_path / "gradle-build.log"
    log_path.write_text("compile failure\nmissing symbol DebugToken\n", encoding="utf-8")
    _FakeRunner.report = _FakeReport(passed=False, log_path=str(log_path))
    runtime = SimpleNamespace(workspace_root=str(tmp_path))

    receipt = _gradle_fallback_receipt(
        runtime,
        tmp_path,
        runtime_module=_runtime_module(),
        jdt_error=RuntimeError("owner unavailable"),
        gradle_runner_factory=_FakeRunner,
    )

    assert receipt["status"] == "FAIL"
    assert receipt["error_count"] == 1
    diagnostic = receipt["diagnostics"][0]
    assert diagnostic["code"] == "GRADLE_BUILD_FAILED"
    assert "missing symbol DebugToken" in diagnostic["message"]
