from __future__ import annotations

from types import SimpleNamespace

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
