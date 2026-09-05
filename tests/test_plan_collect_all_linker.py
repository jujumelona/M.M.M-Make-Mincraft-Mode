from __future__ import annotations

import pytest

from minecraft_mod_ai.plan_collect_all_linker import (
    PlanCollectAllLinkError,
    collect_plan_link_issues,
    validate_plan_collect_all,
)


def _task(task_id: str, *, anchors=(), depends_on=(), gates=(), provides=()):
    return {
        "task_id": task_id,
        "owned_anchors": list(anchors),
        "depends_on": list(depends_on),
        "required_gates": list(gates),
        "provides": list(provides),
        "reuse_refs": [],
    }


def _source(path: str, symbol: str, *, status: str = "host_reserved"):
    return {
        "kind": "symbol",
        "locator": f"{path}#{symbol}",
        "status": status,
        "module_id": "root",
        "source_set": "main",
    }


def _test(path: str, symbol: str):
    return {
        "kind": "test",
        "locator": f"{path}#{symbol}",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "test",
    }


def _binding(task_ref: str, anchor, *, binding_id: str | None = None):
    return {
        "production_module_id": binding_id or f"binding-{task_ref}",
        "task_ref": task_ref,
        "reuse_action": "fresh",
        "reuse_refs": [],
        "owned_anchors": [anchor],
    }


def test_collect_all_reports_every_deterministic_plan_defect_in_one_pass():
    good_source = _source("src/main/java/example/Good.java", "Good")
    unowned_source = _source("src/main/java/example/Other.java", "Other")
    test_only = _test("src/test/java/example/RuntimeTest.java", "RuntimeTest")

    plan = {
        "tasks": [
            _task(
                "missing-binding",
                anchors=(good_source,),
                gates=("target_compile",),
                provides=("capability:missing",),
                depends_on=("unknown-task",),
            ),
            _task(
                "test-only",
                anchors=(test_only,),
                gates=("source_static_validation", "target_compile"),
                provides=("capability:runtime",),
            ),
            _task(
                "unowned-binding",
                anchors=(good_source,),
                gates=("target_compile",),
                provides=("capability:owned",),
            ),
            _task("cycle-a", anchors=(good_source,), depends_on=("cycle-b",)),
            _task("cycle-b", anchors=(good_source,), depends_on=("cycle-a",)),
        ]
    }
    handoff = {
        "production_modules": [
            _binding("unowned-binding", unowned_source),
            _binding("cycle-a", good_source),
            _binding("cycle-b", good_source),
        ],
        "asset_requests": [],
    }

    issues = collect_plan_link_issues(plan, handoff)
    codes = {issue.code for issue in issues}

    assert "TASK_DEPENDENCY_UNKNOWN" in codes
    assert "TASK_EXECUTABLE_BINDING_MISSING" in codes
    assert "TASK_RUNTIME_TEST_ONLY" in codes
    assert "TASK_SOURCE_BINDING_MISSING" in codes
    assert "PRODUCTION_BINDING_NOT_OWNED" in codes
    assert "TASK_DEPENDENCY_CYCLE" in codes
    assert len(issues) >= 6

    with pytest.raises(PlanCollectAllLinkError) as exc_info:
        validate_plan_collect_all(plan, handoff)
    assert len(exc_info.value.issues) == len(issues)
    assert "TASK_EXECUTABLE_BINDING_MISSING" in str(exc_info.value)
    assert "TASK_DEPENDENCY_CYCLE" in str(exc_info.value)


def test_valid_exact_source_and_asset_bindings_pass_collect_all():
    source = _source("src/main/java/example/Feature.java", "Feature")
    resource = {
        "kind": "resource",
        "locator": "src/main/resources/assets/example/lang/en_us.json",
        "status": "host_reserved",
        "module_id": "root",
        "source_set": "main",
    }
    plan = {
        "tasks": [
            _task(
                "source-task",
                anchors=(source,),
                gates=("source_static_validation", "target_compile"),
                provides=("capability:feature",),
            ),
            _task("asset-task", anchors=(resource,), depends_on=("source-task",)),
        ]
    }
    handoff = {
        "production_modules": [_binding("source-task", source)],
        "asset_requests": [
            {
                "asset_request_id": "asset-1",
                "task_ref": "asset-task",
                "reuse_action": "fresh",
                "reuse_refs": [],
                "locator": resource["locator"],
            }
        ],
    }

    assert collect_plan_link_issues(plan, handoff) == ()
    validate_plan_collect_all(plan, handoff)


def test_fresh_binding_with_reuse_refs_is_rejected():
    source = _source("src/main/java/example/Feature.java", "Feature")
    task = _task("feature", anchors=(source,), gates=("target_compile",))
    task["reuse_refs"] = ["component:donor"]
    plan = {"tasks": [task]}
    handoff = {
        "production_modules": [_binding("feature", source)],
        "asset_requests": [],
    }

    codes = {issue.code for issue in collect_plan_link_issues(plan, handoff)}
    assert "TASK_FRESH_HAS_REUSE_REFS" in codes
