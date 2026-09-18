from __future__ import annotations

from types import SimpleNamespace

from minecraft_mod_ai.complete_orchestrator import CompleteProductionOrchestrator


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
