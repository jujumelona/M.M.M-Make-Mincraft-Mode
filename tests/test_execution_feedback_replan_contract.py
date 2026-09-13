from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from minecraft_mod_ai import execution_feedback_replan_contract as feedback
from minecraft_mod_ai.execution_feedback_exception_scope_contract import (
    _checkpoint_for_exception,
)
from minecraft_mod_ai.execution_feedback_owner_precision_contract import (
    install as install_owner_precision,
)


class _FakeLedger:
    def __init__(self, tasks):
        self._tasks = list(tasks)

    def tasks(self, *, cursor="", limit=1000, state=None):
        assert state is None
        assert limit == 1000
        if cursor:
            return {"tasks": [], "next_cursor": ""}
        return {"tasks": self._tasks, "next_cursor": ""}


def _generation_task(node_id: str, module_id: str, path: str, requirement: str):
    return {
        "node_id": node_id,
        "stage": "generate:custom",
        "state": "succeeded",
        "payload": {"members": [{"module_id": module_id}]},
        "receipt": {
            "semantic_observations": [
                {
                    "schema_version": "mmm/semantic-task-observation-v2",
                    "task_id": module_id,
                    "task_ids": [module_id],
                    "requirement_refs": [requirement],
                    "touched_paths": [path],
                }
            ]
        },
    }


def test_feedback_path_selects_only_observed_generation_owner():
    ledger = _FakeLedger(
        [
            _generation_task(
                "generate-custom-00000000",
                "alpha",
                "/workspace/mod/src/main/java/demo/Alpha.java",
                "REQ-ALPHA",
            ),
            _generation_task(
                "generate-custom-00000001",
                "beta",
                "/workspace/mod/src/main/java/demo/Beta.java",
                "REQ-BETA",
            ),
        ]
    )
    seeds, owners, requirements, matches = feedback._derive_impacted_seeds(
        ledger,
        {
            "checkpoint_id": "validate-jdt",
            "diagnostics": [
                {
                    "path": "src/main/java/demo/Alpha.java",
                    "message": "cannot resolve symbol",
                }
            ],
        },
    )

    assert seeds == {"generate-custom-00000000"}
    assert "alpha" in owners
    assert "beta" not in owners
    assert "REQ-ALPHA" in requirements
    assert "REQ-BETA" not in requirements
    assert [item["node_id"] for item in matches] == ["generate-custom-00000000"]


def test_unowned_base_project_diagnostics_route_to_prepare_project():
    ledger = _FakeLedger(
        [
            _generation_task(
                "generate-custom-00000000",
                "debug_token",
                "src/main/java/dev/mmm/debugfixture/DebugToken.java",
                "REQ-DEBUG",
            )
        ]
    )
    diagnostics = [
        {"path": "src/main/resources/fabric.mod.json", "code": "BAD_FABRIC_DEPENDS"},
        {"path": "src/main/resources/pack.mcmeta", "code": "BAD_RESOURCE_PACK_FORMAT"},
        {
            "path": "src/main/resources/assets/mmm_debug_fixture/lang/en_us.json",
            "code": "MISSING_JSON",
        },
        {
            "path": "src/main/resources/assets/mmm_debug_fixture/lang/ko_kr.json",
            "code": "MISSING_JSON",
        },
    ]

    seeds, owners, _requirements, matches = feedback._derive_impacted_seeds(
        ledger,
        {"checkpoint_id": "validate-source", "diagnostics": diagnostics},
    )

    assert seeds == {"prepare-project"}
    assert owners == {"prepare-project"}
    prepare_match = next(item for item in matches if item["node_id"] == "prepare-project")
    assert set(prepare_match["diagnostic_paths"]) == {item["path"] for item in diagnostics}
    assert prepare_match["match"]["host_base_project"] is True


def test_generation_owner_wins_before_base_project_fallback():
    ledger = _FakeLedger(
        [
            _generation_task(
                "generate-content-00000000",
                "metadata_owner",
                "src/main/resources/fabric.mod.json",
                "REQ-META",
            )
        ]
    )

    seeds, owners, requirements, matches = feedback._derive_impacted_seeds(
        ledger,
        {
            "checkpoint_id": "validate-source",
            "diagnostics": [
                {"path": "src/main/resources/fabric.mod.json", "code": "BAD_ENTRYPOINTS"}
            ],
        },
    )

    assert seeds == {"generate-content-00000000"}
    assert "prepare-project" not in seeds
    assert owners == {"metadata_owner"}
    assert requirements == {"REQ-META"}
    assert [item["node_id"] for item in matches] == ["generate-content-00000000"]


def test_path_binding_does_not_use_basename_only():
    assert feedback._path_equivalent(
        "/workspace/mod/src/main/java/a/Widget.java",
        "src/main/java/a/Widget.java",
    )
    assert not feedback._path_equivalent(
        "/workspace/mod/src/main/java/a/Widget.java",
        "/workspace/mod/src/main/java/b/Widget.java",
    )


def test_batched_receipt_ownership_does_not_inherit_positional_member():
    module = SimpleNamespace(module_id="wrong-positional-owner")
    dummy = SimpleNamespace(_receipt_owner_ids=lambda _module, _receipt: [])
    install_owner_precision(dummy)

    owners = dummy._receipt_owner_ids(
        module,
        {
            "schema_version": "mmm/extended-content-v2",
            "modules": ["alpha", "beta"],
        },
    )
    assert owners == ["alpha", "beta"]
    assert "wrong-positional-owner" not in owners


def test_exception_scope_never_reuses_old_validation_for_runtime_failure():
    assert (
        _checkpoint_for_exception(
            RuntimeError("Generated complete project failed deterministic validation.")
        )
        == "validate-source"
    )
    assert (
        _checkpoint_for_exception(
            RuntimeError("JDT reported errors and automatic repair is disabled.")
        )
        == "validate-jdt"
    )
    assert (
        _checkpoint_for_exception(
            RuntimeError("Gradle/GameTest failed after the repair loop.")
        )
        == "gradle-build"
    )
    assert _checkpoint_for_exception(RuntimeError("VisualCritic rejected screenshots")) is None


def test_diagnostics_extract_path_from_gradle_message():
    diagnostics = feedback._diagnostics_from_value(
        {
            "status": "FAIL",
            "build": {
                "error": "src/main/java/demo/Alpha.java:42: cannot find symbol"
            },
        }
    )
    assert any(
        item["path"].endswith("src/main/java/demo/Alpha.java")
        for item in diagnostics
    )


class _FeedbackLoopError(RuntimeError):
    pass


@dataclass(frozen=True)
class _FeedbackLoopOptions:
    resume: bool = False


class _FeedbackLoopLedger:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate_execution_feedback(self, _feedback):
        self.invalidations += 1
        return {
            "feedback_fingerprint": f"feedback-{self.invalidations}",
            "global_replan_required": False,
            "impacted_generation_node_ids": [f"generate-{self.invalidations}"],
        }


def _feedback_loop_module(*, failures_before_success: int | None):
    class Orchestrator:
        def __init__(self) -> None:
            self.calls = 0
            self._mmm_feedback_ledger = _FeedbackLoopLedger()

        def _open_run(self, _run_name, _plan, *, resume):
            return None, self._mmm_feedback_ledger, bool(resume)

        def execute(self, *_args, **_kwargs):
            self.calls += 1
            if failures_before_success is None or self.calls <= failures_before_success:
                raise _FeedbackLoopError(f"failure-{self.calls}")
            return "done"

    return SimpleNamespace(
        CompleteProductionOrchestrator=Orchestrator,
        CompleteProductionError=_FeedbackLoopError,
        CompleteExecutionOptions=_FeedbackLoopOptions,
    )


def _install_feedback_loop(monkeypatch, *, failures_before_success: int | None):
    feedback_rows = []

    def latest_failed_feedback(_ledger):
        feedback_rows.append(len(feedback_rows) + 1)
        return {"checkpoint_id": "validate-source", "diagnostics": []}

    monkeypatch.setattr(feedback, "_latest_failed_feedback", latest_failed_feedback)
    module = _feedback_loop_module(failures_before_success=failures_before_success)
    feedback._install_run_context(module)
    return module.CompleteProductionOrchestrator(), feedback_rows


def test_execution_feedback_repair_allows_more_than_two_distinct_reentries(monkeypatch):
    orchestrator, feedback_rows = _install_feedback_loop(
        monkeypatch, failures_before_success=4
    )

    assert orchestrator.execute() == "done"
    assert orchestrator.calls == 5
    assert orchestrator._mmm_feedback_ledger.invalidations == 4
    assert len(feedback_rows) == 4


def test_execution_feedback_repair_stops_on_repeated_fingerprint(monkeypatch):
    orchestrator, feedback_rows = _install_feedback_loop(
        monkeypatch, failures_before_success=None
    )

    def repeated_feedback(_feedback):
        orchestrator._mmm_feedback_ledger.invalidations += 1
        return {
            "feedback_fingerprint": "same-feedback",
            "global_replan_required": False,
            "impacted_generation_node_ids": ["generate-same"],
        }

    orchestrator._mmm_feedback_ledger.invalidate_execution_feedback = repeated_feedback

    with pytest.raises(_FeedbackLoopError, match="failure-2"):
        orchestrator.execute()
    assert orchestrator.calls == 2
    assert orchestrator._mmm_feedback_ledger.invalidations == 2
    assert len(feedback_rows) == 2
