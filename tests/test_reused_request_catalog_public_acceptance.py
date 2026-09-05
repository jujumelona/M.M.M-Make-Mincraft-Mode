from __future__ import annotations

import pytest

from minecraft_mod_ai import evidence_first_planning as evidence
from minecraft_mod_ai import production_contract as production
from minecraft_mod_ai.production_boundary_contract import install_production_boundary_contract


PROMPT = "Add a weather compass and keep task_internal trace metadata private."
PUBLIC_ACCEPTANCE = "The weather compass reports the observed weather to the player."
INTERNAL_ACCEPTANCE = (
    "Task task_internal: done_predicate verifies all declared provides and owned anchors."
)


def _target() -> dict[str, object]:
    return {
        "target": {
            "minecraft_version": "26.1.2",
            "loader": "fabric",
            "java_version": "25",
            "fabric_loader": "0.18.4",
            "fabric_api": "0.140.2+26.1",
            "fabric_loom": "1.14.10",
            "gradle": "9.2.1",
            "gradle_sha256": "a" * 64,
            "data_pack_version": "101.1",
            "resource_pack_version": "84.0",
            "resource_pack_format": 84,
            "release_metadata_url": (
                "https://piston-meta.mojang.com/v1/packages/deadbeef/26.1.2.json"
            ),
            "source_api_family": "fabric_live_ai",
        }
    }


def _task_modules(plan: dict) -> list[dict[str, object]]:
    return [
        {
            "module_id": task["task_id"],
            "kind": "custom_java",
            "config": {},
            "depends_on": [],
            "required_gates": [],
        }
        for task in plan["tasks"]
    ]


def _clean_plan() -> tuple[dict, dict[str, object]]:
    install_production_boundary_contract()
    design: dict[str, object] = {
        "title": "Weather compass",
        "acceptance_tests": [PUBLIC_ACCEPTANCE],
    }
    plan = evidence.compile_evidence_first_plan(
        PROMPT,
        design,
        target_decision=_target(),
    )
    return plan, design


def _rehash_verified_plan(plan: dict) -> None:
    request = plan["request_catalog"]
    requirements = {
        requirement["requirement_id"]: requirement
        for requirement in request["requirements"]
    }
    for gap in plan["gap_catalog"]:
        requirement = requirements[gap["requirement_ref"]]
        gap["acceptance"] = list(requirement["acceptance"])
        gap["gap_sha256"] = ""
        gap["gap_sha256"] = evidence._hash_without(gap, "gap_sha256")

    rebuilt_tasks = evidence._compile_tasks(
        plan["gap_catalog"],
        plan["reuse_decisions"],
        plan["target_decision"],
        plan["branch_predicates"],
        plan["ownership_context"],
    )
    order = evidence._topological(rebuilt_tasks)
    tasks_by_id = {task["task_id"]: task for task in rebuilt_tasks}
    plan["tasks"] = [tasks_by_id[task_id] for task_id in order]

    request["catalog_sha256"] = ""
    request["catalog_sha256"] = evidence._hash_without(
        request,
        "catalog_sha256",
    )
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = evidence._hash_without(plan, "plan_sha256")
    evidence.validate_evidence_first_plan(plan, prompt=PROMPT)


def test_verified_legacy_plan_migrates_internal_public_acceptance() -> None:
    plan, design = _clean_plan()
    plan["request_catalog"]["requirements"][0]["acceptance"] = [
        INTERNAL_ACCEPTANCE
    ]
    plan["acceptance_release_bindings"][0]["acceptance"] = [INTERNAL_ACCEPTANCE]
    _rehash_verified_plan(plan)

    compiled = production.compile_production_contract(
        requested_prompt=PROMPT,
        game_design=design,
        modules=_task_modules(plan),
        acceptance_tests=[],
        evidence_plan=plan,
    )

    public_statements = [
        item["statement"]
        for item in compiled.contract["acceptance_catalog"]
        if item["visibility"] == "public"
    ]
    assert public_statements
    folded = " ".join(public_statements).casefold()
    for marker in (
        "task_",
        "done_predicate",
        "declared provides",
        "owned anchor",
    ):
        assert marker not in folded


def test_verified_plan_migration_rejects_tampered_hash_before_rewrite() -> None:
    plan, design = _clean_plan()
    plan["acceptance_release_bindings"][0]["acceptance"] = [INTERNAL_ACCEPTANCE]

    with pytest.raises(evidence.EvidencePlanError, match="hash mismatch"):
        production.compile_production_contract(
            requested_prompt=PROMPT,
            game_design=design,
            modules=_task_modules(plan),
            acceptance_tests=[],
            evidence_plan=plan,
        )
