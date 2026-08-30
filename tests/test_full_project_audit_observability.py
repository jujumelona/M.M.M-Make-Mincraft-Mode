from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

from tools import full_project_audit as audit


def _redirect_artifacts(tmp_path: Path, monkeypatch) -> None:
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    monkeypatch.setattr(audit, "AUDIT_DIR", audit_dir)
    monkeypatch.setattr(audit, "LOG_PATH", audit_dir / "FULL_PROJECT_AUDIT.log")
    monkeypatch.setattr(audit, "REPORT_PATH", audit_dir / "FULL_PROJECT_AUDIT.json")
    audit.LOG_PATH.write_text("", encoding="utf-8")


def test_check_payload_has_one_unambiguous_status_semantics() -> None:
    expected = {
        audit.PASS: (True, False, False),
        audit.WARN: (False, False, True),
        audit.SKIP: (False, False, True),
        audit.FAIL: (False, True, False),
    }
    for status, semantics in expected.items():
        payload = audit.Check("probe", status, "detail").payload()
        assert (
            payload["passed"],
            payload["blocking_failure"],
            payload["non_blocking"],
        ) == semantics


def test_unexpected_probe_exception_is_internal_with_traceback_only_in_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    audit.CHECKS.clear()

    def broken_probe() -> None:
        raise TypeError("programming defect")

    audit.isolated_probe("broken_probe", "runtime", broken_probe)
    assert len(audit.CHECKS) == 1
    check = audit.CHECKS[0]
    assert check.status == audit.FAIL
    assert check.category == "audit-internal"
    assert "TypeError: programming defect" in check.detail
    assert "Traceback" not in check.detail

    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert "[INTERNAL TRACEBACK]" in log
    assert "TypeError: programming defect" in log


def test_assertion_probe_failure_remains_validation_not_internal(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    audit.CHECKS.clear()

    def failed_contract() -> None:
        raise AssertionError("contract mismatch")

    audit.isolated_probe("contract_probe", "runtime", failed_contract)
    assert len(audit.CHECKS) == 1
    check = audit.CHECKS[0]
    assert check.status == audit.FAIL
    assert check.category == "runtime"
    assert "AssertionError: contract mismatch" in check.detail
    assert "[INTERNAL TRACEBACK]" not in audit.LOG_PATH.read_text(encoding="utf-8")


def test_logged_process_streams_full_output_but_keeps_bounded_tail(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    line_count = 5000
    result = audit._run_logged_process(
        [
            sys.executable,
            "-c",
            f"for i in range({line_count}): print('row-%05d-' % i + 'x' * 200)",
        ],
        timeout=30,
        cwd=tmp_path,
        label="large-output",
    )
    assert result.returncode == 0
    assert result.error is None
    assert result.timed_out is False
    assert len(result.output_tail) <= audit._OUTPUT_TAIL_LIMIT
    assert "row-04999" in result.output_tail

    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert "row-00000" in log
    assert "row-04999" in log
    assert sum(line.startswith("row-") for line in log.splitlines()) == line_count


def test_streamed_process_log_redacts_environment_secrets(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    monkeypatch.setenv("MMM_TEST_API_KEY", "do-not-leak-this-value")
    result = audit._run_logged_process(
        [sys.executable, "-c", "print('do-not-leak-this-value')"],
        timeout=30,
        cwd=tmp_path,
        label="secret-output",
    )
    assert result.returncode == 0
    assert "do-not-leak-this-value" not in result.output_tail
    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert "do-not-leak-this-value" not in log
    assert "<redacted>" in log


def test_streaming_redactor_masks_secrets_across_every_small_chunk_boundary() -> None:
    exact_secret = "cross-boundary-secret-value"
    source = (
        "prefix/"
        + exact_secret
        + "/middle token="
        + ("v" * 257)
        + "; authorization : Bearer-super-secret /suffix"
    )
    expected = (
        "prefix/<redacted>/middle token=<redacted>; "
        "authorization : <redacted>"
    )

    for width in range(1, 33):
        redactor = audit.StreamingRedactor((exact_secret,))
        chunks = (source[index : index + width] for index in range(0, len(source), width))
        rendered = "".join(redactor.feed(chunk) for chunk in chunks) + redactor.finish()
        assert rendered == expected, f"chunk width {width} broke streaming redaction"
        assert exact_secret not in rendered
        assert "Bearer-super-secret" not in rendered
        assert "v" * 32 not in rendered


def test_streaming_redactor_masks_quoted_structured_and_header_values() -> None:
    source = (
        '{"token": "alpha beta", '
        '"password":"abc\\\" def", '
        '"secret":{"nested":["one two", {"credential":"three"}]}, '
        '"safe": 1}\n'
        "Authorization: Bearer bearer-credential\n"
        "Cookie: session=abc; other=def\n"
        "visible=ok"
    )
    expected = (
        '{"token": "<redacted>", '
        '"password":"<redacted>", '
        '"secret":<redacted>, '
        '"safe": 1}\n'
        "Authorization: <redacted>\n"
        "Cookie: <redacted>\n"
        "visible=ok"
    )

    for width in range(1, 41):
        redactor = audit.StreamingRedactor()
        rendered = "".join(
            redactor.feed(source[index : index + width])
            for index in range(0, len(source), width)
        ) + redactor.finish()
        assert rendered == expected, f"chunk width {width} broke structured redaction"
        for leaked in (
            "alpha beta",
            "abc\\\" def",
            "one two",
            "credential",
            "three",
            "bearer-credential",
            "session=abc",
            "other=def",
        ):
            assert leaked not in rendered


def test_streaming_redactor_does_not_match_sensitive_substrings_in_identifiers() -> None:
    source = "mytoken=value tokenizer=value apikeyring=value token=value"
    expected = "mytoken=value tokenizer=value apikeyring=value token=<redacted>"

    for width in range(1, 17):
        redactor = audit.StreamingRedactor()
        rendered = "".join(
            redactor.feed(source[index : index + width])
            for index in range(0, len(source), width)
        ) + redactor.finish()
        assert rendered == expected, f"chunk width {width} caused a boundary false positive"


def test_streaming_redactor_finish_never_flushes_unclosed_sensitive_payload() -> None:
    sources = (
        'token=unclosed-secret-value',
        '{"token":"unclosed quoted secret',
        '{"secret":{"nested":"unclosed structured secret"',
    )
    for source in sources:
        for width in range(1, 11):
            redactor = audit.StreamingRedactor()
            rendered = "".join(
                redactor.feed(source[index : index + width])
                for index in range(0, len(source), width)
            ) + redactor.finish()
            assert "unclosed" not in rendered
            assert "secret-value" not in rendered
            assert "quoted secret" not in rendered
            assert "structured secret" not in rendered
            assert "<redacted>" in rendered


def test_nonstreaming_redact_uses_same_structured_contract() -> None:
    source = (
        '{"token": "alpha beta", '
        '"password": {"raw": "nested value"}, '
        '"safe": "visible"}'
    )
    rendered = audit.redact(source, secrets=())
    assert rendered == (
        '{"token": "<redacted>", '
        '"password": <redacted>, '
        '"safe": "visible"}'
    )
    assert "alpha beta" not in rendered
    assert "nested value" not in rendered


def test_streamed_process_redacts_quoted_json_without_exact_secret_registration(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "_OUTPUT_CHUNK_CHARS", 5)
    code = "print('{\"token\": \"alpha beta\", \"safe\": 1}')"
    result = audit._run_logged_process(
        [sys.executable, "-c", code],
        timeout=30,
        cwd=tmp_path,
        label="quoted-json-secret-output",
    )

    assert result.returncode == 0
    assert "alpha beta" not in result.output_tail
    assert '"safe": 1' in result.output_tail
    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert "alpha beta" not in log
    assert '"token": "<redacted>"' in log
    assert '"safe": 1' in log


def test_single_unbroken_multimegabyte_line_has_bounded_python_memory(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "_OUTPUT_CHUNK_CHARS", 4096)
    monkeypatch.setattr(audit, "_environment_secret_values", lambda: ())

    raw_path = tmp_path / "single-line.log"
    payload_size = 8 * 1024 * 1024
    with raw_path.open("w", encoding="utf-8") as handle:
        chunk = "x" * 8192
        for _ in range(payload_size // len(chunk)):
            handle.write(chunk)

    tracemalloc.start()
    try:
        tail = audit._drain_process_output(raw_path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(tail) <= audit._OUTPUT_TAIL_LIMIT
    assert audit.LOG_PATH.stat().st_size == payload_size
    assert peak < 2 * 1024 * 1024, f"streaming peak memory regressed: {peak} bytes"


def test_no_newline_process_output_masks_cross_chunk_label_and_exact_secret(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "_OUTPUT_CHUNK_CHARS", 17)
    exact_secret = "do-not-leak-cross-chunk"
    monkeypatch.setenv("MMM_TEST_SECRET", exact_secret)

    code = (
        "import sys; "
        "sys.stdout.write('A' * 15 + 'do-not-leak-cross-chunk' + "
        "' tok' + 'en=' + 'z' * 2000000 + ';done')"
    )
    result = audit._run_logged_process(
        [sys.executable, "-c", code],
        timeout=30,
        cwd=tmp_path,
        label="single-line-secret-output",
    )

    assert result.returncode == 0
    assert result.error is None
    assert result.timed_out is False
    assert exact_secret not in result.output_tail
    assert "z" * 32 not in result.output_tail
    assert "done" in result.output_tail

    log = audit.LOG_PATH.read_text(encoding="utf-8")
    assert exact_secret not in log
    assert "z" * 32 not in log
    assert "token=<redacted>;done" in log


def test_timeout_preserves_output_tail_without_memory_capture(
    tmp_path: Path, monkeypatch
) -> None:
    _redirect_artifacts(tmp_path, monkeypatch)
    result = audit._run_logged_process(
        [
            sys.executable,
            "-c",
            "import time; print('before-timeout', flush=True); time.sleep(2)",
        ],
        timeout=0.05,
        cwd=tmp_path,
        label="timeout-output",
    )
    assert result.timed_out is True
    assert result.returncode is None
    assert "before-timeout" in result.output_tail
    assert "process_timeout=" in audit.LOG_PATH.read_text(encoding="utf-8")
