"""Recorded failure replay and real compiler-to-linker boundary regressions."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import complete_planner
from minecraft_mod_ai.acceptance_contracts import is_public_acceptance
from minecraft_mod_ai.evidence_first_planning import (
    _compile_tasks,
    compile_evidence_first_plan,
)
from minecraft_mod_ai.evidence_execution_contract import (
    _execution_task,
    _validate_derived_owners,
    execution_plan,
    execution_handoff,
)
from minecraft_mod_ai.evidence_first_handoff import (
    _production_modules_for_task,
    _asset_requests_for_task,
    build_evidence_first_handoff,
)
from minecraft_mod_ai.plan_collect_all_linker import (
    collect_plan_link_issues,
    PlanCollectAllLinkError,
)
from minecraft_mod_ai.research_derived_requirements import (
    derive_research_requirements,
    attach_derived_requirement_ledger,
)
from minecraft_mod_ai.research_requirement_plan_slice import host_facet_baseline
from tests.planning_authority_fixtures import request_catalog


def _recording():
    return json.loads(
        (Path(__file__).parent / "fixtures/planner_failure_2026_09_05.json").read_text()
    )


def _lower_and_link(tasks, ownership):
    lowered = [
        _execution_task({"ownership_context": ownership}, task) for task in tasks
    ]
    handoff = {
        "production_modules": [
            item.to_dict()
            for task in lowered
            for item in _production_modules_for_task(
                task, action="fresh", reuse_refs=()
            )
        ],
        "asset_requests": [
            item.to_dict()
            for task in lowered
            for item in _asset_requests_for_task(task, action="fresh", reuse_refs=())
        ],
    }
    return lowered, handoff


def _reachability(tasks):
    by_id = {task["task_id"]: task for task in tasks}

    # Stable semantic exports survive regenerated task identifiers and index changes.
    def identity(task):
        return tuple(task["requirement_refs"]), tuple(task["provides"])

    result = {}
    for task in tasks:
        seen = set()
        todo = list(task["depends_on"])
        while todo:
            ref = todo.pop()
            if ref not in seen:
                seen.add(ref)
                todo.extend(by_id[ref]["depends_on"])
        result[identity(task)] = {identity(by_id[ref]) for ref in seen}
    return result


def test_recorded_gate_failure_is_removed_without_losing_any_dependency():
    fixture = _recording()
    compiler = fixture["compiler"]
    tasks = list(_compile_tasks(**compiler, emit_trace=False))
    assert len(fixture["recorded_tasks"]) == 161
    assert len(tasks) == 142  # Seven fake gates and twelve inherited template steps removed.
    assert not any(
        "requirement_ready:" in value for task in tasks for value in task["provides"]
    )
    current, original = _reachability(tasks), _reachability(fixture["recorded_tasks"])
    assert len(current) == len(tasks)
    assert set(current) <= set(original)
    for task, predecessors in current.items():
        assert predecessors == original[task] & current.keys()
    lowered, handoff = _lower_and_link(tasks, compiler["ownership"])
    assert collect_plan_link_issues({"tasks": lowered}, handoff) == ()
    for gap in compiler["gaps"]:
        requirement = dict(gap, requirement_id=gap["requirement_ref"])
        baseline = host_facet_baseline(
            requirement,
            [
                task
                for task in tasks
                if gap["requirement_ref"] in task["requirement_refs"]
            ],
        )
        assert all(item["disposition"] != "missing" for item in baseline.values())


def test_old_gates_and_runtime_with_test_only_binding_still_fail_closed():
    fixture = _recording()
    tasks, handoff = _lower_and_link(
        fixture["recorded_tasks"], fixture["compiler"]["ownership"]
    )
    gates = {
        task["task_id"]
        for task in tasks
        if any(value.startswith("requirement_ready:") for value in task["provides"])
    }
    assert len(gates) == 7

    # The retired prerequisite-gate tasks are deliberately not promoted into fake Java
    # runtime tasks. Even after execution lowering they therefore remain unbound and the
    # deterministic linker rejects all seven instead of manufacturing source ownership.
    lowered_issues = collect_plan_link_issues({"tasks": tasks}, handoff)
    missing_bindings = {
        issue.task_ref
        for issue in lowered_issues
        if issue.code == "TASK_EXECUTABLE_BINDING_MISSING"
    }
    assert gates <= missing_bindings

    raw_issues = collect_plan_link_issues({"tasks": fixture["recorded_tasks"]}, {})
    raw_missing = {
        issue.task_ref
        for issue in raw_issues
        if issue.code == "TASK_EXECUTABLE_BINDING_MISSING"
    }
    assert gates <= raw_missing

    runtime = copy.deepcopy(
        next(
            task
            for task in tasks
            if any(v.startswith("capability:") for v in task["provides"])
        )
    )
    runtime["owned_anchors"] = [
        anchor for anchor in runtime["owned_anchors"] if anchor["kind"] == "test"
    ]
    runtime["depends_on"] = []
    runtime["execution_role"] = (
        "verification"  # An asserted role cannot bypass capability validation.
    )
    codes = {issue.code for issue in collect_plan_link_issues({"tasks": [runtime]}, {})}
    assert "TASK_EXECUTABLE_BINDING_MISSING" in codes
    assert "TASK_RUNTIME_TEST_ONLY" in codes


def _design(lock, prompt, requirements):
    return {
        "_platform_selection": {
            "source": "platform_resolver",
            "target": lock.to_dict(),
        },
        "modules": [],
        "acceptance_tests": ["Trading is atomic."],
        "_evidence_request_catalog": request_catalog(prompt, requirements),
    }


class _NoModel:
    def generate_text(self, *_args, **_kwargs):
        raise AssertionError("No external evidence should need a model call")


def test_real_plan_lowering_and_handoff_before_and_after_research(
    synthetic_platform_lock,
):
    prompt = "Add trade. Add quests."
    design = _design(
        synthetic_platform_lock,
        prompt,
        [
            {
                "requirement_id": "req_trade",
                "capability": "economy.trade",
                "source_text": "Add trade.",
                "statement": "Add atomic player trading.",
            },
            {
                "requirement_id": "req_quests",
                "capability": "quest.progression",
                "source_text": "Add quests.",
                "statement": "Add persistent quest progression.",
            },
        ],
    )
    plan = compile_evidence_first_plan(prompt, design)
    for candidate in (
        plan,
        attach_derived_requirement_ledger(
            plan,
            derive_research_requirements(
                _NoModel(),
                prompt=prompt,
                evidence_plan=plan,
                research_brief={},
                technical_evidence={},
                game_design=design,
            ),
        ),
    ):
        lowered = execution_plan(candidate)
        handoff = execution_handoff(
            candidate, build_evidence_first_handoff(candidate), lowered
        )
        assert not collect_plan_link_issues(lowered, handoff)
        batches = complete_planner._evidence_host_batches(candidate)
        assert {batch.batch_id for batch in batches} == {
            task["task_id"] for task in candidate["tasks"]
        }


def test_deterministic_link_failure_precedes_any_augmentation(monkeypatch):
    artifacts = SimpleNamespace(
        game_design={}, research_brief={}, technical_evidence={}
    )
    monkeypatch.setattr(
        complete_planner,
        "PlanningPipeline",
        lambda _router: SimpleNamespace(prepare=lambda *_a, **_k: artifacts),
    )
    monkeypatch.setattr(
        complete_planner, "compile_evidence_first_plan", lambda *_a: {"tasks": []}
    )

    def reject(_plan):
        raise PlanCollectAllLinkError(())

    monkeypatch.setattr(complete_planner, "_evidence_host_batches", reject)
    monkeypatch.setattr(
        complete_planner,
        "derive_research_requirements",
        lambda *_a, **_k: pytest.fail("late preflight"),
    )
    with pytest.raises(PlanCollectAllLinkError):
        complete_planner.CompleteGameDesignPlanner(_NoModel()).plan("Add trade.")


def test_derived_obligation_is_bound_to_exactly_one_valid_parent_task():
    task = {
        "task_id": "owner",
        "requirement_refs": ["req_a"],
        "owned_anchors": [],
        "acceptance": [],
    }
    sibling = dict(task, task_id="sibling")
    decision = {
        "disposition": "derived",
        "parent_requirement_ref": "req_a",
        "owner_task_ref": "owner",
        "implementation_obligations": ["Reject duplicate transfer"],
        "acceptance": ["Exactly one debit"],
    }
    plan = {"derived_requirement_ledger": {"facet_decisions": [decision]}}
    _validate_derived_owners(plan, [task, sibling])
    assert _execution_task(plan, task)["implementation_obligations"] == [
        "Reject duplicate transfer"
    ]
    assert _execution_task(plan, sibling)["implementation_obligations"] == []
    for owner in ("", "unknown"):
        decision["owner_task_ref"] = owner
        with pytest.raises(ValueError, match="execution owner"):
            _validate_derived_owners(plan, [task, sibling])
    decision["owner_task_ref"] = "owner"
    decision["parent_requirement_ref"] = "req_other"
    with pytest.raises(ValueError, match="does not own requirement"):
        _validate_derived_owners(plan, [task, sibling])


def test_real_host_batches_compile_production_contract_without_rewriting_semantics(
    synthetic_platform_lock,
):
    from minecraft_mod_ai import production_contract

    prompt = "Add a persistent networked machine with a GUI and generated resources."
    design = _design(
        synthetic_platform_lock,
        prompt,
        [
            {
                "requirement_id": "req_machine",
                "capability": "automation.machine",
                "source_text": prompt,
                "statement": prompt,
                "implementation_capabilities": [
                    "automation.machine",
                    "persistence.state_store",
                    "network.action_sync",
                    "ui.container",
                ],
            }
        ],
    )
    plan = compile_evidence_first_plan(prompt, design)
    original = copy.deepcopy(plan)
    planner = complete_planner.CompleteGameDesignPlanner(_NoModel())
    modules, assets, checks = planner._expand_batches(
        complete_planner._evidence_host_batches(plan),
        evidence_mode=True,
        evidence_acceptance_tests=[
            check
            for requirement in plan["request_catalog"]["requirements"]
            for check in requirement["acceptance"]
        ],
    )
    compiled = production_contract.compile_production_contract(
        requested_prompt=prompt,
        game_design={"pitch": prompt},
        modules=modules,
        assets=assets,
        acceptance_tests=checks,
        evidence_plan=plan,
    )
    assert compiled.contract
    assert plan == original
    assert set(compiled.acceptance_tests) == {
        check for check in checks if is_public_acceptance(check)
    }
    assert {module.module_id for module in modules} == {
        task["task_id"] for task in plan["tasks"]
    }
