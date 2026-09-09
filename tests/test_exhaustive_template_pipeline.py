from __future__ import annotations

from collections import Counter

import pytest

from minecraft_mod_ai.fixed_pipeline_definition import (
    ExecutionLedger,
    FIXED_PIPELINE,
    StageExecution,
    run_exhaustive_pipeline,
)
from minecraft_mod_ai.minecraft_template_catalog import profile_for_capability
from minecraft_mod_ai.minecraft_template_steps import (
    execute_all_template_families,
    family_execution_ledger,
    family_ids,
    steps_for_profile,
)


def test_fixed_pipeline_is_large_unique_and_exhaustive_across_all_major_phases() -> None:
    FIXED_PIPELINE.validate()
    ids = [stage.stage_id for stage in FIXED_PIPELINE.stages]
    counts = Counter(stage.phase for stage in FIXED_PIPELINE.stages)

    assert len(ids) == len(set(ids))
    assert len(ids) >= 2600
    assert FIXED_PIPELINE.definition_sha256.startswith("sha256:")
    assert len(FIXED_PIPELINE.definition_sha256) == 71
    assert counts["capture"] >= 12
    assert counts["intent"] >= 28
    assert counts["domain"] >= 26
    assert counts["requirements"] >= 500
    assert counts["reuse"] >= 170
    assert counts["planning"] >= 30
    assert counts["architecture"] >= 150
    assert counts["implementation"] >= 800
    assert counts["assets"] >= 400
    assert counts["resources"] >= 200
    assert counts["linking"] >= 25
    assert counts["validation"] >= 40
    assert counts["repair"] >= 100
    assert counts["packaging"] >= 20


def test_every_template_family_is_executed_even_when_not_applicable() -> None:
    profile = profile_for_capability("item.weapon")
    executions = execute_all_template_families(profile)
    expected = family_ids()

    assert tuple(item.family_id for item in executions) == expected
    assert len(executions) == len(expected)
    assert all(item.executed is True for item in executions)
    assert sum(item.status == "required" for item in executions) == 1
    assert sum(item.status == "not_applicable" for item in executions) == len(expected) - 1

    ledger = family_execution_ledger(profile)
    assert ledger["total"] == len(expected)
    assert ledger["executed"] == len(expected)
    assert ledger["unexecuted"] == 0
    assert ledger["failed"] == 0
    assert ledger["complete"] is True


def test_compatibility_steps_api_uses_exhaustive_execution_result() -> None:
    profile = profile_for_capability("network.transaction")
    executions = execute_all_template_families(profile)
    required_steps = tuple(
        step
        for execution in executions
        if execution.status == "required"
        for step in execution.steps
    )

    assert steps_for_profile(profile) == required_steps
    assert required_steps
    assert profile.capability in required_steps[-1].provides


def test_ledger_cannot_complete_with_an_unexecuted_stage() -> None:
    ledger = ExecutionLedger(FIXED_PIPELINE)
    for stage in FIXED_PIPELINE.stages[:-1]:
        ledger.record(StageExecution(stage_id=stage.stage_id, status="not_applicable"))

    summary = ledger.summary()
    assert summary["executed"] == len(FIXED_PIPELINE.stages) - 1
    assert summary["unexecuted"] == 1
    assert summary["failed"] == 0
    assert summary["project_complete"] is False


def test_ledger_cannot_complete_with_a_failed_stage() -> None:
    failed_stage = FIXED_PIPELINE.stages[len(FIXED_PIPELINE.stages) // 2].stage_id

    def executor(stage, _context):
        return "fail" if stage.stage_id == failed_stage else "not_applicable"

    ledger = run_exhaustive_pipeline({}, executor)
    summary = ledger.summary()
    assert summary["executed"] == len(FIXED_PIPELINE.stages)
    assert summary["unexecuted"] == 0
    assert summary["failed"] == 1
    assert summary["project_complete"] is False


def test_executor_failure_does_not_short_circuit_remaining_templates() -> None:
    crashed_stage = FIXED_PIPELINE.stages[17].stage_id
    visited: list[str] = []

    def executor(stage, _context):
        visited.append(stage.stage_id)
        if stage.stage_id == crashed_stage:
            raise RuntimeError("synthetic stage crash")
        return "not_required"

    ledger = run_exhaustive_pipeline({}, executor)

    assert visited == [stage.stage_id for stage in FIXED_PIPELINE.stages]
    assert len(ledger.records) == len(FIXED_PIPELINE.stages)
    assert ledger.summary()["failed"] == 1
    crash = next(record for record in ledger.records if record.stage_id == crashed_stage)
    assert crash.status == "fail"
    assert crash.details["error_type"] == "RuntimeError"


def test_all_success_terminal_records_are_required_for_project_complete() -> None:
    ledger = run_exhaustive_pipeline({}, lambda _stage, _context: "not_applicable")
    summary = ledger.summary()

    assert summary["executed"] == len(FIXED_PIPELINE.stages)
    assert summary["unexecuted"] == 0
    assert summary["failed"] == 0
    assert summary["project_complete"] is True
    assert summary["pipeline_sha256"] == FIXED_PIPELINE.definition_sha256


def test_ledger_rejects_duplicate_and_unknown_execution_records() -> None:
    stage = FIXED_PIPELINE.stages[0]
    ledger = ExecutionLedger(FIXED_PIPELINE)
    ledger.record(StageExecution(stage_id=stage.stage_id, status="pass"))

    with pytest.raises(ValueError, match="more than once"):
        ledger.record(StageExecution(stage_id=stage.stage_id, status="pass"))
    with pytest.raises(ValueError, match="unknown fixed stage"):
        ledger.record(StageExecution(stage_id="UNKNOWN-999", status="pass"))
