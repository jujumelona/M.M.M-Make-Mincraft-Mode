from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _jdt_release_evidence_passed,
    _requested_verification_failures,
    _run_release_jdt_verification,
)
from minecraft_mod_ai.complete_orchestrator_support import file_sha256
from minecraft_mod_ai.java_core import JavaCoreService


def _clean_jdt_core_receipt():
    """Mirror the identity-bearing JDT Core receipt, not the retired LSP shape."""
    return {
        "schema_version": "mmm/java-diagnostics-v3",
        "verification_backend": "jdt_core",
        "verification_scope": "full",
        "complete": True,
        "diagnostics": {},
        "error_count": 0,
        "model_id": "fixture-model-id",
        "model_revision": "fixture-model-revision",
        "session_id": "fixture-session-id",
    }


def test_absent_optional_jdt_evidence_does_not_emit_a_failure(monkeypatch):
    import minecraft_mod_ai.validation_diagnostic_contract as diagnostics

    events = []
    monkeypatch.setattr(diagnostics, "emit_root_cause", lambda *a, **kw: events.append(kw))
    assert not _jdt_release_evidence_passed(None)
    assert events == []
    assert _requested_verification_failures(run_jdt=True, jdt_receipt=None) == [
        "execution-gate:jdt:missing-jdt"
    ]


def _proposal():
    module = SimpleNamespace(
        module_id="debug_token",
        required_gates=("target_compile",),
        kind="custom_java",
        config={},
    )
    return SimpleNamespace(
        modules=(module,),
        base_proposal=SimpleNamespace(spec=SimpleNamespace()),
    )


def _failures(build_report):
    return CompleteProductionOrchestrator._required_gate_failures(
        _proposal(),
        generated_receipts=(),
        project_root=None,
        source_validation={"status": "PASS"},
        jdt_receipt={"status": "UNAVAILABLE", "error_count": 0, "files_opened": 0},
        build_report=build_report,
        jar_validation=None,
        blockbench_receipts=(),
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )


def test_target_compile_requires_real_successful_gradle_command_receipt():
    assert _failures(
        {
            "status": "PASS",
            "commands": [
                {"name": "build", "exit_code": 0, "timed_out": False},
            ],
        }
    ) == []


def test_target_compile_rejects_status_only_without_gradle_command_receipt():
    failures = _failures({"status": "PASS", "commands": []})
    assert failures == [
        "required-gate:debug_token:target_compile:missing-gradle"
    ]


def test_target_compile_rejects_timed_out_or_failed_gradle_command():
    for command in (
        {"name": "build", "exit_code": 1, "timed_out": False},
        {"name": "clean_build", "exit_code": 0, "timed_out": True},
    ):
        failures = _failures({"status": "PASS", "commands": [command]})
        assert failures == [
            "required-gate:debug_token:target_compile:missing-gradle"
        ]


def _jdt_failures(jdt_receipt, *, build_report=None):
    module = SimpleNamespace(
        module_id="debug_token",
        required_gates=("jdt",),
        kind="custom_java",
        config={},
    )
    proposal = SimpleNamespace(
        modules=(module,),
        base_proposal=SimpleNamespace(spec=SimpleNamespace()),
    )
    return CompleteProductionOrchestrator._required_gate_failures(
        proposal,
        generated_receipts=(),
        project_root=None,
        source_validation={"status": "PASS"},
        jdt_receipt=jdt_receipt,
        build_report=build_report,
        jar_validation=None,
        blockbench_receipts=(),
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )


def test_required_jdt_does_not_fallback_to_successful_gradle_build():
    failures = _jdt_failures(
        {"status": "UNAVAILABLE", "error_count": 0, "files_opened": 0},
        build_report={
            "status": "PASS",
            "commands": [{"name": "build", "exit_code": 0, "timed_out": False}],
        },
    )

    assert failures == ["required-gate:debug_token:jdt:missing-jdt"]


def test_required_jdt_accepts_only_real_clean_jdt_receipt():
    assert _jdt_failures(
        _clean_jdt_core_receipt(),
        build_report={
            "status": "PASS",
            "commands": [{"name": "build", "exit_code": 0, "timed_out": False}],
        },
    ) == []


def test_run_jdt_option_blocks_release_when_jdt_is_unavailable():
    receipt = {
        "status": "UNAVAILABLE",
        "error": "JDTLanguageServerError: ServiceReady was not observed before validation",
        "diagnostics": {},
        "error_count": 0,
        "files_opened": 0,
    }

    assert not _jdt_release_evidence_passed(receipt)
    assert _requested_verification_failures(
        run_jdt=True,
        jdt_receipt=receipt,
    ) == ["execution-gate:jdt:missing-jdt"]


def test_run_jdt_option_accepts_real_clean_jdt_receipt():
    receipt = _clean_jdt_core_receipt()

    assert _jdt_release_evidence_passed(receipt)
    assert _requested_verification_failures(
        run_jdt=True,
        jdt_receipt=receipt,
    ) == []


def test_run_jdt_option_is_independent_from_proposal_required_gates():
    assert _requested_verification_failures(
        run_jdt=False,
        jdt_receipt=None,
    ) == []
    assert _requested_verification_failures(
        run_jdt=True,
        jdt_receipt=None,
    ) == ["execution-gate:jdt:missing-jdt"]


def test_jdt_release_evidence_unwraps_reviewed_transport_envelope():
    receipt = {"structured_content": _clean_jdt_core_receipt()}

    assert _jdt_release_evidence_passed(receipt)


def test_failed_jdt_core_status_cannot_pass_with_warning_only_diagnostics():
    # A failed status with warning-only diagnostics does not create a severity-1
    # diagnostic. The release gate must still reject the failed verifier state.
    for status in ("FAILED", "ERROR", "UNAVAILABLE", "AVAILABLE", "OK", "DEFERRED_TO_POST_BUILD"):
        receipt = _clean_jdt_core_receipt()
        receipt["status"] = status
        receipt["diagnostics"] = {
            "file:///Example.java": [
                {"severity": 2, "message": "warning only"}
            ]
        }
        assert not _jdt_release_evidence_passed(receipt), status

    successful = _clean_jdt_core_receipt()
    successful["status"] = "PASS"
    assert _jdt_release_evidence_passed(successful)


def test_jdt_release_evidence_rejects_deferred_postbuild_state():
    receipt = {
        "status": "DEFERRED_TO_POST_BUILD",
        "diagnostics": {},
        "error_count": 0,
        "files_opened": 1,
    }

    assert not _jdt_release_evidence_passed(receipt)


def test_generated_receipt_cannot_add_unapproved_release_gate():
    proposal = _proposal()
    failures = CompleteProductionOrchestrator._required_gate_failures(
        proposal,
        generated_receipts=(
            {
                "module_id": "debug_token",
                "required_gates": [
                    "Blockbench UV and bone hierarchy review",
                    "restart persistence test",
                ],
            },
        ),
        project_root=None,
        source_validation={"status": "PASS"},
        jdt_receipt={"status": "UNAVAILABLE", "error_count": 0, "files_opened": 0},
        build_report={
            "status": "PASS",
            "commands": [{"name": "build", "exit_code": 0, "timed_out": False}],
        },
        jar_validation=None,
        blockbench_receipts=(),
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )

    assert failures == []


def test_blockbench_required_gate_uses_entity_review_receipt(tmp_path):
    preview = tmp_path / "entity-preview.png"
    preview.write_bytes(b"png")
    module = SimpleNamespace(
        module_id="boss_dragon",
        required_gates=("Blockbench UV and bone hierarchy review",),
        kind="boss",
        config={},
    )
    proposal = SimpleNamespace(
        modules=(module,),
        base_proposal=SimpleNamespace(spec=SimpleNamespace()),
    )

    failures = CompleteProductionOrchestrator._required_gate_failures(
        proposal,
        generated_receipts=(),
        project_root=None,
        source_validation={"status": "PASS"},
        jdt_receipt=None,
        build_report=None,
        jar_validation=None,
        blockbench_receipts=(
            {
                "entity": "boss_dragon",
                "uv": {"status": "PASS"},
                "preview": str(preview),
                "preview_sha256": file_sha256(preview),
            },
        ),
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )

    assert failures == []


def test_legacy_lsp_receipt_cannot_satisfy_release_jdt_gate():
    receipt = {
        "schema_version": "mmm/java-diagnostics-v2",
        "status": "PASS",
        "diagnostics": {},
        "error_count": 0,
        "files_opened": 2,
    }
    assert not _jdt_release_evidence_passed(receipt)
    assert _jdt_failures(receipt) == [
        "required-gate:debug_token:jdt:missing-jdt"
    ]


def test_jdt_core_receipt_requires_complete_identity_and_clean_diagnostics():
    for field in ("model_id", "model_revision", "session_id"):
        receipt = _clean_jdt_core_receipt()
        del receipt[field]
        assert not _jdt_release_evidence_passed(receipt), field

    receipt = _clean_jdt_core_receipt()
    receipt["complete"] = False
    assert not _jdt_release_evidence_passed(receipt)

    receipt = _clean_jdt_core_receipt()
    receipt["verification_backend"] = "legacy_lsp"
    assert not _jdt_release_evidence_passed(receipt)

    receipt = _clean_jdt_core_receipt()
    receipt["error_count"] = 1
    receipt["diagnostics"] = {
        "file:///Example.java": [
            {"severity": 1, "code": "COMPILER_ERROR", "message": "invalid source"}
        ]
    }
    assert not _jdt_release_evidence_passed(receipt)


def test_release_jdt_calls_jdt_core_once_with_requested_timeout(monkeypatch, tmp_path):
    calls = []
    expected = _clean_jdt_core_receipt()

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        "minecraft_mod_ai.complete_orchestrator.run_jdt_diagnostics",
        fake_run,
    )
    receipt = _run_release_jdt_verification(tmp_path, timeout_seconds=7)
    assert receipt is expected
    assert calls == [((JavaCoreService, tmp_path), {"timeout_seconds": 7})]


def test_release_jdt_preserves_unavailable_without_unreviewed_retries(monkeypatch, tmp_path):
    calls = []
    expected = {
        "status": "UNAVAILABLE",
        "error": "OwnerRPCError: no project JDK matching Java 25",
        "diagnostics": {},
        "error_count": 0,
    }

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(
        "minecraft_mod_ai.complete_orchestrator.run_jdt_diagnostics",
        fake_run,
    )
    receipt = _run_release_jdt_verification(tmp_path, timeout_seconds=7)
    assert receipt is expected
    assert not _jdt_release_evidence_passed(receipt)
    assert calls == [((JavaCoreService, tmp_path), {"timeout_seconds": 7})]
