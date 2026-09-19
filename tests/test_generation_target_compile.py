from __future__ import annotations

from minecraft_mod_ai import generation_target_compile as target_compile


TARGET = "src/main/java/demo/Test.java"


class _Build:
    def __init__(self, report):
        self.report = report

    def to_dict(self):
        return dict(self.report)


class _Runner:
    def __init__(self, report):
        self.report = report

    def build(self, project_root, *, run_gametest):
        assert run_gametest is False
        return _Build(self.report)


def _install_runner(monkeypatch, report):
    monkeypatch.setattr(
        target_compile,
        "GradleRunner",
        lambda _cache: _Runner(report),
    )


def test_generation_target_compile_passes_only_real_build_pass(tmp_path, monkeypatch) -> None:
    _install_runner(
        monkeypatch,
        {
            "status": "PASS",
            "commands": [{"name": "build", "exit_code": 0, "timed_out": False}],
        },
    )
    monkeypatch.setattr(
        target_compile,
        "compiler_log_diagnostics",
        lambda _build, *, project_root: [],
    )

    receipt = target_compile.run_generation_target_compile(
        tmp_path,
        target_path=TARGET,
    )

    assert receipt["status"] == "PASS"
    assert receipt["reason"] == "target compiler passed"


def test_generation_target_compile_fails_on_owned_javac_diagnostic(
    tmp_path,
    monkeypatch,
) -> None:
    _install_runner(
        monkeypatch,
        {
            "status": "FAIL",
            "commands": [{"name": "build", "exit_code": 1, "timed_out": False}],
        },
    )
    monkeypatch.setattr(
        target_compile,
        "compiler_log_diagnostics",
        lambda _build, *, project_root: [
            {"path": TARGET, "message": "cannot find symbol"}
        ],
    )

    receipt = target_compile.run_generation_target_compile(
        tmp_path,
        target_path=TARGET,
    )

    assert receipt["status"] == "FAIL"
    assert len(receipt["diagnostics"]) == 1


def test_generation_target_compile_defers_unowned_build_failure(
    tmp_path,
    monkeypatch,
) -> None:
    _install_runner(
        monkeypatch,
        {
            "status": "FAIL",
            "commands": [{"name": "build", "exit_code": 1, "timed_out": False}],
        },
    )
    monkeypatch.setattr(
        target_compile,
        "compiler_log_diagnostics",
        lambda _build, *, project_root: [
            {
                "path": "src/main/java/demo/Other.java",
                "message": "cannot find symbol",
            }
        ],
    )

    receipt = target_compile.run_generation_target_compile(
        tmp_path,
        target_path=TARGET,
    )

    assert receipt["status"] == "DEFERRED"
    assert receipt["diagnostics"] == []


def test_generation_target_compile_reports_infrastructure_unavailable(
    tmp_path,
    monkeypatch,
) -> None:
    class _BrokenRunner:
        def build(self, project_root, *, run_gametest):
            raise OSError("compiler unavailable")

    monkeypatch.setattr(
        target_compile,
        "GradleRunner",
        lambda _cache: _BrokenRunner(),
    )
    monkeypatch.setattr(
        target_compile,
        "compiler_log_diagnostics",
        lambda _build, *, project_root: [],
    )

    receipt = target_compile.run_generation_target_compile(
        tmp_path,
        target_path=TARGET,
    )

    assert receipt["status"] == "UNAVAILABLE"
    assert "compiler unavailable" in receipt["reason"]


def test_generation_target_compile_rejects_non_java_target(tmp_path) -> None:
    try:
        target_compile.run_generation_target_compile(
            tmp_path,
            target_path="src/main/resources/value.json",
        )
    except ValueError as exc:
        assert "Java target" in str(exc)
    else:
        raise AssertionError("non-Java target_compile request must fail")
