from __future__ import annotations

import copy
import json

import pytest

from minecraft_mod_ai.complete_planner import _evidence_host_batches
from minecraft_mod_ai.evidence_first_planning import (
    EvidencePlanError,
    _hash_without,
    _sha,
    build_request_catalog,
    compile_evidence_first_plan,
    validate_evidence_first_plan,
)
from minecraft_mod_ai.minecraft_template_catalog import requirement_branch_features
from minecraft_mod_ai.platform_catalog import adapter_for_target
from minecraft_mod_ai.project_inventory import inspect_project_inventory


def _request_catalog(prompt: str, *capabilities: str) -> dict[str, object]:
    requirements: list[dict[str, object]] = []
    for index, raw_capability in enumerate(capabilities, 1):
        capability = str(raw_capability).removeprefix("capability:")
        acceptance = f"The {capability} behavior is observable in Minecraft."
        requirements.append(
            {
                "requirement_id": f"req_{index:03d}",
                "capability": capability,
                "statement": prompt,
                "semantic_statement": prompt,
                "mandatory": True,
                "provenance_role": "explicit_test_authority",
                "source_span": {
                    "source_id": "requested_prompt",
                    "char_start": 0,
                    "char_end": len(prompt),
                    "text": prompt,
                    "text_sha256": _sha(prompt),
                },
                "evidence_refs": [],
                "derived_from": [],
                "depends_on": [],
                "provides": [f"capability:{capability}"],
                "gameplay_capabilities": [capability],
                "implementation_capabilities": [capability],
                "implementation_obligations": [
                    f"Implement the grounded {capability} capability and verify its observable behavior."
                ],
                "artifact_task_ids": [],
                "semantic_type": "gameplay_mechanic",
                "unlock_policy": {},
                "artifact_obligations": [],
                "design_resolution_obligations": [],
                "runtime_acceptance": [acceptance],
                "semantic_status": "RESOLVED",
                "unresolved_spans": [],
                "acceptance": [acceptance],
                "observable_behavior": {
                    "given": "the grounded requirement preconditions are established",
                    "when": prompt,
                    "then": acceptance,
                },
                "template_profile": {
                    "template_id": "explicit_test_authority",
                    "architecture_owner": "host",
                },
                "search_queries": [],
                "reuse_candidates": [],
                "detailed_plan_ref": f"detail_{index:03d}",
                "engineering_worksheet": None,
            }
        )

    catalog: dict[str, object] = {
        "prompt_sha256": _sha(prompt),
        "prompt_char_length": len(prompt),
        "purpose": prompt,
        "requirements": requirements,
        "constraints": [],
        "non_goals": [],
        "deployment_expectations": [],
        "requirement_graph": {
            "node_ids": [item["requirement_id"] for item in requirements],
            "edges": [],
        },
        "dependency_provenance": [],
        "semantic_audit": {
            "status": "APPROVED",
            "generation_policy": "explicit_test_authority",
        },
        "planning_state_sha256": "test-fixture",
        "catalog_sha256": "",
    }
    catalog["catalog_sha256"] = _hash_without(catalog, "catalog_sha256")
    return catalog


def _design(prompt: str, *capabilities: str) -> dict[str, object]:
    return {
        "pitch": "Implement only the requested behavior.",
        "modules": [
            {"plugin_id": capability, "reason": capability}
            for capability in capabilities
        ],
        "acceptance_tests": [
            f"The {capability} behavior is observable in Minecraft."
            for capability in capabilities
        ],
        "_evidence_request_catalog": _request_catalog(prompt, *capabilities),
        "_platform_selection": {
            "target": adapter_for_target("1.21.1", "fabric").public_dict(),
            "preserved_existing_target": True,
            "migration_requested": False,
        },
    }


def _rehash(plan: dict[str, object]) -> None:
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = _hash_without(plan, "plan_sha256")


def _public_acceptance(plan: dict[str, object]) -> list[str]:
    return [
        str(item)
        for binding in plan["acceptance_release_bindings"]
        for item in binding["acceptance"]
    ]


def test_plan_hash_and_semantic_ids_are_deterministic() -> None:
    prompt = "Add a machine with saved state, synced packets, and a screen."
    capabilities = ("automation.machine", "network.action_sync", "ui.menu")
    first = compile_evidence_first_plan(prompt, _design(prompt, *capabilities))
    second = compile_evidence_first_plan(prompt, _design(prompt, *capabilities))
    assert first == second
    assert first["plan_sha256"].startswith("sha256:")
    assert len(first["request_catalog"]["requirements"]) == 3
    assert len(first["tasks"]) > 10


def test_prompt_only_catalog_requires_grounded_planning_authority() -> None:
    with pytest.raises(EvidencePlanError, match="PLANNING_STATE_AUTHORITY_REQUIRED"):
        build_request_catalog("Add a machine.", {})


def test_design_module_name_cannot_add_semantic_authority() -> None:
    prompt = "Add quests."
    design = {
        "modules": [{"plugin_id": "placeholder", "reason": "placeholder"}],
        "_platform_selection": _design(prompt, "quest.state")["_platform_selection"],
    }
    with pytest.raises(EvidencePlanError, match="PLANNING_STATE_AUTHORITY_REQUIRED"):
        build_request_catalog(prompt, design)


def test_frozen_catalog_is_reused_without_design_module_augmentation() -> None:
    prompt = "Add trade. Add quests."
    catalog = _request_catalog(prompt, "trade.transaction")
    design = {
        **_design(prompt, "trade.transaction"),
        "modules": [{"plugin_id": "placeholder", "reason": "placeholder"}],
        "_evidence_request_catalog": catalog,
    }
    reused = build_request_catalog(prompt, design)
    assert [item["capability"] for item in reused["requirements"]] == ["trade.transaction"]


def test_machine_vertical_dag_uses_exact_provider_edges() -> None:
    prompt = "Add a machine with saved state, synced packets, and a screen."
    plan = compile_evidence_first_plan(
        prompt,
        _design(prompt, "automation.machine", "network.action_sync", "ui.menu"),
    )
    branches = plan["branch_predicates"]
    assert branches["needs_registry"]["status"] == "ACTIVE"
    assert branches["needs_persistence"]["status"] == "ACTIVE"
    assert branches["needs_network"]["status"] == "ACTIVE"
    assert branches["needs_client_render"]["status"] == "ACTIVE"
    provider = {
        provided: task["task_id"]
        for task in plan["tasks"]
        for provided in task["provides"]
    }
    roots = set(plan["root_provides"])
    for task in plan["tasks"]:
        assert set(task["depends_on"]) == {
            provider[consumed] for consumed in task["consumes"] if consumed not in roots
        }


def test_self_claimed_external_component_never_removes_exact_semantic_gap() -> None:
    prompt = "Add trade."
    components = [
        {
            "component_id": "existing_trade",
            "kind": "symbol",
            "locator": "src/main/java/example/TradeService.java#TradeService",
            "content_sha256": "sha256:" + "a" * 64,
            "provides": ["capability:trade.transaction"],
            "requires": [],
            "bound_to_project": True,
            "verification_status": "verified",
            "provenance": {
                "origin": "external",
                "repository": "https://example.invalid/repository",
                "commit_sha": "a" * 40,
            },
        }
    ]
    plan = compile_evidence_first_plan(
        prompt,
        _design(prompt, "trade.transaction"),
        component_catalog=components,
    )
    assert plan["reuse_decisions"][0]["action"] == "fresh"
    assert plan["reuse_decisions"][0]["evidence_status"] == "not_applicable"
    assert plan["gap_catalog"]


def test_scanner_attested_project_component_can_close_matching_capability(tmp_path) -> None:
    prompt = "Add trade service."
    src = tmp_path / "src/main/java/example"
    src.mkdir(parents=True)
    (src / "TradeService.java").write_text(
        "package example;\npublic final class TradeService {\n  public void trade() {}\n}\n",
        encoding="utf-8",
    )
    inventory = inspect_project_inventory(tmp_path)
    assert any("capability:trade_service" in component.provides for component in inventory.components)
    design = _design(prompt, "trade_service")
    design["_existing_project_inventory"] = inventory.to_dict()
    plan = compile_evidence_first_plan(prompt, design)
    assert plan["reuse_decisions"][0]["action"] == "retain"
    assert not plan["gap_catalog"]


def test_matching_catalog_requirement_can_bind_several_templates_without_duplicate_semantics() -> None:
    prompt = "Add a persisted networked machine."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "automation.machine"))
    requirement = plan["request_catalog"]["requirements"][0]
    features = requirement_branch_features(requirement)
    assert "needs_persistence" in features
    assert "needs_network" in features
    assert len(plan["request_catalog"]["requirements"]) == 1


def test_validator_rejects_extra_task_or_missing_task_hash() -> None:
    prompt = "Add trade."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "trade.transaction"))
    bad = copy.deepcopy(plan)
    bad["tasks"].append(copy.deepcopy(bad["tasks"][0]))
    _rehash(bad)
    with pytest.raises(EvidencePlanError):
        validate_evidence_first_plan(bad)
    bad = copy.deepcopy(plan)
    bad["tasks"][0]["task_sha256"] = ""
    _rehash(bad)
    with pytest.raises(EvidencePlanError):
        validate_evidence_first_plan(bad)


def test_unknown_capability_is_preserved_and_owned_by_deterministic_task() -> None:
    prompt = "Add a bespoke temporal resonance mechanic."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "temporal.resonance"))
    assert plan["request_catalog"]["requirements"][0]["capability"] == "temporal.resonance"
    assert any(
        "capability:temporal.resonance" in task["provides"]
        for task in plan["tasks"]
    )


def test_release_acceptance_is_behavioral_not_internal_plan_checks() -> None:
    prompt = "Add trade."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "trade.transaction"))
    assert all(_evidence_host_batches(plan))
    acceptance = _public_acceptance(plan)
    assert acceptance
    assert all("task_" not in item for item in acceptance)
    assert all("owned anchors" not in item.casefold() for item in acceptance)


def test_top_level_design_acceptance_cannot_override_grounded_requirement_acceptance() -> None:
    prompt = "Add trade."
    design = _design(prompt, "trade.transaction")
    design["acceptance_tests"] = ["task_x: verify owned anchors"]
    plan = compile_evidence_first_plan(prompt, design)
    acceptance = _public_acceptance(plan)
    assert "task_x: verify owned anchors" not in acceptance
    assert "The trade.transaction behavior is observable in Minecraft." in acceptance


def test_unknown_requirement_dependency_is_rejected() -> None:
    prompt = "Add trade."
    design = _design(prompt, "trade.transaction")
    design["_evidence_request_catalog"]["requirements"][0]["depends_on"] = ["req_missing"]
    design["_evidence_request_catalog"]["catalog_sha256"] = _hash_without(
        design["_evidence_request_catalog"], "catalog_sha256"
    )
    with pytest.raises(EvidencePlanError):
        compile_evidence_first_plan(prompt, design)


def test_plan_serializes_without_non_json_types() -> None:
    prompt = "Add trade."
    json.dumps(compile_evidence_first_plan(prompt, _design(prompt, "trade.transaction")), ensure_ascii=False)


def test_request_catalog_preserves_explicit_requirement_order() -> None:
    prompt = "A then B then C."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "a.one", "b.two", "c.three"))
    assert [item["capability"] for item in plan["request_catalog"]["requirements"]] == [
        "a.one", "b.two", "c.three"
    ]


def test_root_capabilities_are_unique() -> None:
    prompt = "Add trade and quests."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "trade.transaction", "quest.state"))
    assert len(plan["root_provides"]) == len(set(plan["root_provides"]))


def test_plan_hash_changes_when_platform_target_changes() -> None:
    prompt = "Add trade."
    first_design = _design(prompt, "trade.transaction")
    second_design = _design(prompt, "trade.transaction")
    second_design["_platform_selection"] = {
        "target": adapter_for_target("1.20.1", "fabric").public_dict(),
        "preserved_existing_target": False,
        "migration_requested": True,
    }
    first = compile_evidence_first_plan(prompt, first_design)
    second = compile_evidence_first_plan(prompt, second_design)
    assert first["plan_sha256"] != second["plan_sha256"]


def test_resolved_generation_plan_requires_platform_target() -> None:
    prompt = "Add trade."
    design = _design(prompt, "trade.transaction")
    design.pop("_platform_selection")
    with pytest.raises(EvidencePlanError, match="Target decision is unresolved"):
        compile_evidence_first_plan(prompt, design)


def test_plan_rejects_tampered_target_decision_hash() -> None:
    prompt = "Add trade."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "trade.transaction"))
    bad = copy.deepcopy(plan)
    bad["target_decision"]["decision_sha256"] = "sha256:" + "0" * 64
    _rehash(bad)
    with pytest.raises(EvidencePlanError, match="Target decision hash mismatch"):
        validate_evidence_first_plan(bad)


def test_request_catalog_hash_covers_source_span() -> None:
    prompt = "Add trade."
    catalog = _request_catalog(prompt, "trade.transaction")
    original = catalog["catalog_sha256"]
    catalog["requirements"][0]["source_span"]["text"] = "Add quests."
    assert _hash_without(catalog, "catalog_sha256") != original


def test_request_catalog_rejects_synthetic_semantics() -> None:
    prompt = "Add trade."
    with pytest.raises(EvidencePlanError, match="UNRESOLVED_SEMANTICS"):
        compile_evidence_first_plan(prompt, _design(prompt, "custom.semantic_deadbeef"))


def test_release_bindings_cover_every_grounded_requirement_even_if_design_summary_is_partial() -> None:
    prompt = "Add trade and quests."
    design = _design(prompt, "trade.transaction", "quest.state")
    design["acceptance_tests"] = ["The trade.transaction behavior is observable in Minecraft."]
    plan = compile_evidence_first_plan(prompt, design)
    bindings = plan["acceptance_release_bindings"]
    assert {item["requirement_ref"] for item in bindings} == {"req_001", "req_002"}
    assert all(item["acceptance"] for item in bindings)


def test_direct_validator_rejects_corrupted_host_plan_before_model_use() -> None:
    prompt = "Add trade."
    plan = compile_evidence_first_plan(prompt, _design(prompt, "trade.transaction"))
    plan["tasks"][0]["task_sha256"] = ""
    with pytest.raises(EvidencePlanError):
        validate_evidence_first_plan(plan)
