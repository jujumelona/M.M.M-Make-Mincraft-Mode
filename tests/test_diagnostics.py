import pytest

from minecraft_mod_ai.diagnostics import (
    DiagnosticCollector,
    FailureCategory,
    FailureEvent,
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


def test_sanitizer_applies_to_message_debug_traceback_and_dedup_key() -> None:
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
            deduplication_key="provider:super-secret",
        )
    payload = collector.to_dicts(include_debug=True)[0]
    assert "super-secret" not in str(payload)
    assert "<redacted>" in str(payload)


def test_final_status_never_uses_ambiguous_pass_for_unresolved_failure() -> None:
    assert "PASS" not in {status.value for status in FailureStatus}


def test_explicit_deduplication_key_groups_varying_retry_messages() -> None:
    collector = DiagnosticCollector()
    for attempt, message in enumerate(("timeout after 1.0s", "timeout after 2.0s"), start=1):
        collector.record(
            FailureEvent(
                stage="network",
                operation="fetch metadata",
                category=FailureCategory.TRANSIENT,
                cause_type="TimeoutError",
                cause=message,
                retryable=attempt == 1,
                final_status=(
                    FailureStatus.DEGRADED if attempt == 1 else FailureStatus.UNAVAILABLE
                ),
                fallback="retry" if attempt == 1 else "cached metadata",
                deduplication_key="metadata-timeout",
            )
        )
    group = collector.groups()[0]
    assert group.attempts == 2
    assert group.event.cause == "timeout after 1.0s"
    assert group.event.retryable is False
    assert group.event.final_status is FailureStatus.UNAVAILABLE
    assert group.fallbacks == ["retry", "cached metadata"]


def test_compact_rendering_bounds_distinct_roots_and_multiline_causes() -> None:
    collector = DiagnosticCollector()
    for index in range(4):
        collector.record(
            FailureEvent(
                stage=f"stage-{index}\nforged-line",
                operation="op",
                category=FailureCategory.INTERNAL,
                cause_type="RuntimeError",
                cause=("line one\nline two " + "x" * 200),
                retryable=False,
                final_status=FailureStatus.FAILED,
            )
        )
    rendered = render_failure_summary(collector.groups(), max_groups=2, text_limit=64)
    assert rendered.count("ROOT FAILURE ") == 2
    assert "ROOT FAILURES OMITTED\n2 additional distinct root failures" in rendered
    assert "line one line two" in rendered
    assert "stage-0 forged-line" in rendered
    assert "x" * 100 not in rendered


def test_rendering_rejects_unbounded_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_groups"):
        render_failure_summary((), max_groups=0)
    with pytest.raises(ValueError, match="text_limit"):
        render_failure_summary((), text_limit=1)
