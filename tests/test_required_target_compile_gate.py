from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import (
    CompleteProductionOrchestrator,
    _jdt_release_evidence_passed,
    _run_release_jdt_verification,
    _requested_verification_failures,
)


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
        {
            "schema_version": "mmm/java-diagnostics-v2",
            "status": "PASS",
            "diagnostics": {},
            "error_count": 0,
            "files_opened": 1,
        },
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
    receipt = {
        "schema_version": "mmm/java-diagnostics-v2",
        "diagnostics": {},
        "error_count": 0,
        "files_opened": 1,
    }

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
    receipt = {
        "structured_content": {
            "schema_version": "mmm/java-diagnostics-v2",
            "diagnostics": {},
            "error_count": 0,
            "files_opened": 2,
        }
    }

    assert _jdt_release_evidence_passed(receipt)


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
                "preview_sha256": CompleteProductionOrchestrator._file_hash(preview),
            },
        ),
        runtime_receipt=None,
        playtest_receipt=None,
        visual_receipt=None,
    )

    assert failures == []


def test_release_jdt_retries_transient_service_ready_once(monkeypatch, tmp_path):
    calls = []
    receipts = iter([
        {
            "status": "UNAVAILABLE",
            "error": "JDTWorkspaceBootstrapError: ServiceReady was not observed before validation",
            "diagnostics": {},
            "error_count": 0,
            "files_opened": 0,
        },
        {
            "schema_version": "mmm/java-diagnostics-v2",
            "status": "PASS",
            "diagnostics": {},
            "error_count": 0,
            "files_opened": 2,
        },
    ])
    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return next(receipts)
    monkeypatch.setattr(
        "minecraft_mod_ai.complete_orchestrator.run_jdt_diagnostics",
        fake_run,
    )
    receipt = _run_release_jdt_verification(tmp_path, timeout_seconds=7, attempts=2)
    assert receipt["verification_attempts"] == 2
    assert receipt["files_opened"] == 2
    assert len(calls) == 2


def test_release_jdt_does_not_retry_nontransient_unavailable(monkeypatch, tmp_path):
    calls = []
    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "status": "UNAVAILABLE",
            "error": "JDTWorkspaceBootstrapError: no project JDK matching Java 25",
            "diagnostics": {},
            "error_count": 0,
            "files_opened": 0,
        }
    monkeypatch.setattr(
        "minecraft_mod_ai.complete_orchestrator.run_jdt_diagnostics",
        fake_run,
    )
    receipt = _run_release_jdt_verification(tmp_path, timeout_seconds=7, attempts=3)
    assert receipt["verification_attempts"] == 1
    assert len(calls) == 1
