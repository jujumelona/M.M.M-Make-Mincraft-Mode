from pathlib import Path

from tools.pytest_diagnostics import failure_groups_from_junit, render_junit_failure_summary


def _write_junit(path: Path, cases: str) -> None:
    path.write_text(f'<testsuite tests="2" failures="2">{cases}</testsuite>', encoding="utf-8")


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
