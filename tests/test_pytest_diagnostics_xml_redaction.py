from __future__ import annotations

from pathlib import Path

from tools import pytest_diagnostics


def test_junit_redaction_preserves_xml_declaration_with_short_secret(
    tmp_path: Path,
    monkeypatch,
) -> None:
    junit = tmp_path / "pytest.xml"
    monkeypatch.setenv("MMM_TEST_SECRET", "8")
    junit.write_text(
        '<?xml version="1.0" encoding="utf-8"?>'
        '<testsuite tests="1" failures="1">'
        '<testcase classname="tests.secret" name="one">'
        '<failure message="token=8">trace 8</failure>'
        '</testcase>'
        '</testsuite>',
        encoding="utf-8",
    )

    pytest_diagnostics._redact_file_in_place(
        junit,
        replacement=pytest_diagnostics._XML_REDACTION_MARKER,
        preserve_xml_declaration=True,
    )

    text = junit.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="utf-8"?>')
    assert 'token=8' not in text
    assert 'token=&lt;redacted&gt;' in text

    analysis = pytest_diagnostics.analyze_junit(junit)
    assert analysis.total == 1
    assert analysis.failed == 1
    assert analysis.errors == 0
