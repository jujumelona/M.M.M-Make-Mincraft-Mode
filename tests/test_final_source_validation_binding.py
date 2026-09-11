from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionError,
    _refresh_validation_after_build,
)


def test_final_validation_reuses_prebuild_evidence_when_manifest_is_unchanged():
    calls = []
    prebuild = {"status": "PASS", "checks_run": 7, "marker": "prebuild"}
    source, jdt, refreshed = _refresh_validation_after_build(
        prebuild_manifest="sha256:same",
        final_manifest="sha256:same",
        source_report=prebuild,
        jdt_receipt={"status": "PASS", "marker": "jdt-prebuild"},
        validate_source=lambda: calls.append("source") or {"status": "PASS"},
        validate_jdt=lambda: calls.append("jdt") or {"status": "PASS"},
    )
    assert source is prebuild
    assert jdt == {"status": "PASS", "marker": "jdt-prebuild"}
    assert refreshed is False
    assert calls == []


def test_final_validation_reruns_source_and_jdt_when_repair_changes_manifest():
    calls = []
    source, jdt, refreshed = _refresh_validation_after_build(
        prebuild_manifest="sha256:before",
        final_manifest="sha256:after",
        source_report={"status": "PASS", "marker": "stale"},
        jdt_receipt={"status": "PASS", "marker": "stale-jdt"},
        validate_source=lambda: calls.append("source") or {"status": "PASS", "marker": "final"},
        validate_jdt=lambda: calls.append("jdt") or {"status": "PASS", "marker": "final-jdt"},
    )
    assert source == {"status": "PASS", "marker": "final"}
    assert jdt == {"status": "PASS", "marker": "final-jdt"}
    assert refreshed is True
    assert calls == ["source", "jdt"]


def test_final_validation_fails_closed_if_repaired_tree_is_invalid():
    try:
        _refresh_validation_after_build(
            prebuild_manifest="sha256:before",
            final_manifest="sha256:after",
            source_report={"status": "PASS"},
            jdt_receipt=None,
            validate_source=lambda: {"status": "FAIL", "errors": ["bad final source"]},
            validate_jdt=None,
        )
    except CompleteProductionError as exc:
        assert "final repaired project failed deterministic validation" in str(exc).lower()
    else:
        raise AssertionError("post-repair validation failure was accepted")
