from __future__ import annotations

import subprocess
from types import SimpleNamespace
from pathlib import Path

from tools import pytest_diagnostics
from tools.pytest_diagnostics import (
    analyze_junit,
    failure_groups_from_junit,
    render_junit_failure_summary,
)


def _write_junit(path: Path, cases: str, *, tests: int = 2, failures: int = 2) -> None:
    path.write_text(
        f'<testsuite tests="{tests}" failures="{failures}">{cases}</testsuite>',
        encoding="utf-8",
    )


def _write_passing_junit(path: Path) -> None:
    _write_junit(
        path,
        '<testcase classname="tests.a" name="one" />',
        tests=1,
        failures=0,
    )


def test_duplicate_junit_failures_collapse_to_one_root_cause(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    _write_junit(
        path,
        '<testcase classname="tests.a" name="one"><failure message="same dependency failure">trace A</failure></testcase>'
        '<testcase classname="tests.b" name="two"><failure message="same dependency failure">trace B</failure></testcase>',
    )
    groups, affected = failure_groups_from_junit(path)
    assert len(groups) == 1
    assert groups[0].attempts == 2
    assert affected[groups[0].event.fingerprint] == ["tests.a::one", "tests.b::two"]
    rendered = render_junit_failure_summary(path)
    assert rendered.count("ROOT FAILURE") == 1
    assert "ATTEMPTS\n2" in rendered
    assert "tests.a::one" in rendered
    assert "tests.b::two" in rendered
    assert "trace A" not in rendered
    assert "trace B" not in rendered


def test_distinct_junit_causes_remain_distinct(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    _write_junit(
        path,
        '<testcase classname="tests.a" name="one"><failure message="first cause">trace</failure></testcase>'
        '<testcase classname="tests.b" name="two"><error message="second cause">trace</error></testcase>',
    )
    groups, _ = failure_groups_from_junit(path)
    assert len(groups) == 2
    rendered = render_junit_failure_summary(path)
    assert rendered.count("ROOT FAILURE") == 2
    assert "first cause" in rendered
    assert "second cause" in rendered


def test_streaming_junit_parser_handles_namespaces_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuite xmlns="urn:junit">'
        '<testcase classname="tests.a" name="pass" />'
        '<testcase classname="tests.a" name="skip"><skipped /></testcase>'
        '<testcase classname="tests.a" name="fail"><failure message="boom">trace</failure></testcase>'
        '<testcase classname="tests.a" name="error"><error message="broken">trace</error></testcase>'
        '</testsuite>',
        encoding="utf-8",
    )
    analysis = analyze_junit(path)
    assert (analysis.total, analysis.failed, analysis.errors, analysis.skipped) == (4, 1, 1, 1)
    assert len(analysis.groups) == 2


def test_main_never_reuses_stale_junit_or_raw_log(tmp_path: Path, monkeypatch, capsys) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"
    log.write_text("stale pytest output\n", encoding="utf-8")
    _write_passing_junit(junit)

    def fake_run(*args, **kwargs):
        assert not junit.exists(), "stale JUnit must be removed before pytest starts"
        assert log.read_text(encoding="utf-8") == ""
        assert kwargs["stdout"] is not subprocess.PIPE
        kwargs["stdout"].write("pytest crashed before producing JUnit\n")
        return SimpleNamespace(returncode=5)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    result = pytest_diagnostics.main(
        ["--log", str(log), "--junit", str(junit), "tests/test_missing.py"]
    )
    assert result == 5
    output = capsys.readouterr().out
    assert "MissingJUnit" in output
    assert "FINAL STATUS\nPASS" not in output
    assert log.read_text(encoding="utf-8").startswith("pytest crashed")


def test_success_without_junit_fails_closed(tmp_path: Path, monkeypatch, capsys) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write("pytest claimed success without proof\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    assert (
        pytest_diagnostics.main(
            ["--log", str(log), "--junit", str(junit), "tests/test_anything.py"]
        )
        == 1
    )
    assert "MissingJUnit" in capsys.readouterr().out


def test_success_with_failure_nodes_fails_closed(tmp_path: Path, monkeypatch, capsys) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write("contradictory pytest run\n")
        _write_junit(
            junit,
            '<testcase classname="tests.a" name="one"><failure message="boom">trace</failure></testcase>',
            tests=1,
            failures=1,
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    assert (
        pytest_diagnostics.main(
            ["--log", str(log), "--junit", str(junit), "tests/test_anything.py"]
        )
        == 1
    )
    assert "PytestExitMismatch" in capsys.readouterr().out


def test_empty_success_junit_is_not_accepted_as_test_evidence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write("no tests somehow returned zero\n")
        junit.write_text('<testsuite tests="0" failures="0" />', encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    assert (
        pytest_diagnostics.main(
            ["--log", str(log), "--junit", str(junit), "tests/test_anything.py"]
        )
        == 1
    )
    assert "EmptyJUnit" in capsys.readouterr().out


def test_output_path_collision_is_rejected_before_pytest(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "same.file"
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    assert (
        pytest_diagnostics.main(
            ["--log", str(output), "--junit", str(output), "tests/test_anything.py"]
        )
        == 2
    )
    assert called is False
    assert "OutputPathCollision" in capsys.readouterr().out


def test_nonpositive_timeout_is_rejected_before_pytest(tmp_path: Path, monkeypatch, capsys) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"
    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    assert (
        pytest_diagnostics.main(
            [
                "--log",
                str(log),
                "--junit",
                str(junit),
                "--timeout-seconds",
                "0",
                "tests/test_anything.py",
            ]
        )
        == 2
    )
    assert called is False
    output = capsys.readouterr().out
    assert "InvalidPytestTimeout" in output
    assert "[INPUT]" in output


def test_zero_durations_omits_pytest_all_durations_mode(tmp_path: Path, monkeypatch) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"
    seen_command: list[str] = []

    def fake_run(command, **kwargs):
        seen_command.extend(command)
        kwargs["stdout"].write("pass\n")
        _write_passing_junit(junit)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    assert (
        pytest_diagnostics.main(
            [
                "--log",
                str(log),
                "--junit",
                str(junit),
                "--durations",
                "0",
                "tests/test_anything.py",
            ]
        )
        == 0
    )
    assert not any(item.startswith("--durations=") for item in seen_command)


def test_junit_summary_redacts_labelled_and_environment_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "junit.xml"
    exact_secret = "do-not-leak-junit-secret"
    monkeypatch.setenv("MMM_TEST_SECRET", exact_secret)
    _write_junit(
        path,
        (
            '<testcase classname="tests.secret" name="one">'
            f'<failure message="token=label-secret {exact_secret}">trace</failure>'
            "</testcase>"
        ),
        tests=1,
        failures=1,
    )

    rendered = render_junit_failure_summary(path)
    assert "label-secret" not in rendered
    assert exact_secret not in rendered
    assert "token=<redacted>" in rendered


def test_main_redacts_pytest_log_and_junit_artifacts(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"
    exact_secret = "do-not-leak-artifact-secret"
    monkeypatch.setenv("MMM_TEST_SECRET", exact_secret)

    def fake_run(*args, **kwargs):
        kwargs["stdout"].write(f"token=raw-secret {exact_secret}\n")
        _write_junit(
            junit,
            (
                '<testcase classname="tests.secret" name="one">'
                f'<failure message="token=xml-secret {exact_secret}">trace</failure>'
                "</testcase>"
            ),
            tests=1,
            failures=1,
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    result = pytest_diagnostics.main(
        ["--log", str(log), "--junit", str(junit), "tests/test_anything.py"]
    )

    assert result == 1
    console = capsys.readouterr().out
    log_text = log.read_text(encoding="utf-8")
    junit_text = junit.read_text(encoding="utf-8")
    for leaked in ("raw-secret", "xml-secret", exact_secret):
        assert leaked not in console
        assert leaked not in log_text
        assert leaked not in junit_text
    assert "token=<redacted>" in log_text
    assert "token=&lt;redacted&gt;" in junit_text
    assert "<redacted>" in console

    analysis = analyze_junit(junit)
    assert analysis.total == 1
    assert analysis.failed == 1
    assert analysis.errors == 0
    assert analysis.groups[0].event.cause == "token=<redacted>"


def test_timeout_returns_124_and_preserves_only_redacted_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    log = tmp_path / "pytest.log"
    junit = tmp_path / "pytest.xml"

    def fake_run(command, **kwargs):
        kwargs["stdout"].write("token=timeout-secret\n")
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    monkeypatch.setattr(pytest_diagnostics.subprocess, "run", fake_run)
    result = pytest_diagnostics.main(
        [
            "--log",
            str(log),
            "--junit",
            str(junit),
            "--timeout-seconds",
            "1",
            "tests/test_anything.py",
        ]
    )

    assert result == 124
    console = capsys.readouterr().out
    assert "TimeoutExpired" in console
    assert "[TRANSIENT]" in console
    assert "timeout-secret" not in console
    assert "timeout-secret" not in log.read_text(encoding="utf-8")
    assert "token=<redacted>" in log.read_text(encoding="utf-8")
