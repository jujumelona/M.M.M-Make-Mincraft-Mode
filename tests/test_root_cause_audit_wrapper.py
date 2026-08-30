from tools.root_cause_audit_wrapper import failure_groups_from_report, render_report_summary


def test_duplicate_failed_check_is_collapsed_to_one_root_with_attempt_count() -> None:
    report = {
        "summary": {"passed": 1, "warned": 0, "failed": 2, "skipped": 0},
        "checks": [
            {"name": "provider", "category": "network", "status": "FAIL", "detail": "timeout"},
            {"name": "provider", "category": "network", "status": "FAIL", "detail": "timeout"},
        ],
    }
    groups = failure_groups_from_report(report)
    assert len(groups) == 1
    assert groups[0].attempts == 2
    rendered = render_report_summary(report)
    assert rendered.count("ROOT FAILURE") == 1
    assert "ATTEMPTS\n2" in rendered
    assert "CHECKS pass=1 warn=0 fail=2 skip=0" in rendered


def test_warning_is_not_mislabeled_as_pass() -> None:
    report = {
        "summary": {"passed": 3, "warned": 1, "failed": 0, "skipped": 2},
        "checks": [],
    }
    assert render_report_summary(report).startswith("FINAL STATUS\nWARN\n")


def test_clean_report_is_explicit_pass() -> None:
    report = {
        "summary": {"passed": 4, "warned": 0, "failed": 0, "skipped": 1},
        "checks": [],
    }
    assert render_report_summary(report).startswith("FINAL STATUS\nPASS\n")
