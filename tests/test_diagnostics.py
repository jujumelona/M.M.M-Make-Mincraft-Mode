from minecraft_mod_ai.diagnostics import (
    DiagnosticCollector,
    FailureCategory,
    FailureStatus,
    render_failure_summary,
)


def test_repeated_failure_is_grouped_without_duplicate_root_failure() -> None:
    collector = DiagnosticCollector()
    for _ in range(3):
        try:
            raise TimeoutError("provider timed out")
        except TimeoutError as exc:
            collector.record_exception(
                exc,
                stage="research",
                operation="fetch source",
                category=FailureCategory.TRANSIENT,
                retryable=True,
                final_status=FailureStatus.UNAVAILABLE,
                fallback="cached source",
            )
    groups = collector.groups()
    assert len(groups) == 1
    assert groups[0].attempts == 3
    rendered = render_failure_summary(groups)
    assert rendered.count("ROOT FAILURE") == 1
    assert "ATTEMPTS\n3" in rendered
    assert "Traceback" not in rendered


def test_internal_failure_keeps_debug_traceback_out_of_user_summary() -> None:
    collector = DiagnosticCollector()
    try:
        raise TypeError("bad internal contract")
    except TypeError as exc:
        collector.record_exception(
            exc,
            stage="planner",
            operation="decode accepted state",
            category=FailureCategory.INTERNAL,
            retryable=False,
            final_status=FailureStatus.FAILED,
        )
    group = collector.groups()[0]
    assert group.event.debug_traceback is not None
    assert "TypeError" in group.to_dict(include_debug=True)["debug_traceback"]
    assert "debug_traceback" not in group.to_dict(include_debug=False)
    assert "Traceback" not in render_failure_summary((group,))


def test_sanitizer_applies_to_message_and_debug_traceback() -> None:
    collector = DiagnosticCollector()
    try:
        raise RuntimeError("token=super-secret")
    except RuntimeError as exc:
        collector.record_exception(
            exc,
            stage="ci",
            operation="probe",
            category=FailureCategory.INTERNAL,
            retryable=False,
            final_status=FailureStatus.FAILED,
            sanitize=lambda value: value.replace("super-secret", "<redacted>"),
        )
    payload = collector.to_dicts(include_debug=True)[0]
    assert "super-secret" not in str(payload)
    assert "<redacted>" in str(payload)


def test_final_status_never_uses_ambiguous_pass_for_unresolved_failure() -> None:
    assert "PASS" not in {status.value for status in FailureStatus}
