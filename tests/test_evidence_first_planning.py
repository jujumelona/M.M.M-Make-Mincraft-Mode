from __future__ import annotations

import copy
import json
import re

import pytest

from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _evidence_host_batches
from minecraft_mod_ai.evidence_first_planning import (
    EvidencePlanError,
    _hash_without,
    _sha,
    build_request_catalog,
    compile_evidence_first_plan,
    validate_evidence_first_plan,
)
from minecraft_mod_ai.minecraft_template_catalog import requirement_branch_features
from minecraft_mod_ai.project_inventory import inspect_project_inventory


def _request_catalog(prompt: str, *capabilities: str) -> dict[str, object]:
    """Build explicit test authority without reviving raw-prompt semantic inference."""

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
            "target": {
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "java_version": 21,
            },
            "preserved_existing_target": True,
            "migration_requested": False,
        },
    }


def _rehash(plan: dict[str, object]) -> None:
    plan["plan_sha256"] = ""
    plan["plan_sha256"] = _hash_without(plan, "plan_sha256")


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
    prompt = "Add a machine with saved state, synced packets, and a screen."

    with pytest.raises(EvidencePlanError, match="PLANNING_STATE_AUTHORITY_REQUIRED"):
        build_request_catalog(prompt, {})


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

    assert [item["capability"] for item in reused["requirements"]] == [
        "trade.transaction"
    ]
    assert reused["catalog_sha256"] == _hash_without(reused, "catalog_sha256")


def test_machine_vertical_dag_uses_exact_provider_edges() -> None:
    prompt = "Add a machine with saved state, synced packets, and a screen."
    capabilities = ("automation.machine", "network.action_sync", "ui.menu")
    plan = compile_evidence_first_plan(prompt, _design(prompt, *capabilities))

    branches = plan["branch_predicates"]
    assert branches["needs_registry"]["status"] == "ACTIVE"
    assert branches["needs_persistence"]["status"] == "ACTIVE"
    assert branches["needs_network"]["status"] == "ACTIVE"
    assert branches["needs_client_render"]["status"] == "ACTIVE"
    assert branches["needs_worldgen"]["status"] == "NOT_APPLICABLE"
    assert branches["needs_mixin"]["status"] == "NOT_APPLICABLE"

    provider = {
        provided: task["task_id"]
        for task in plan["tasks"]
        for provided in task["provides"]
    }
    roots = set(plan["root_provides"])
    for task in plan["tasks"]:
        expected = {
            provider[consumed]
            for consumed in task["consumes"]
            if consumed not in roots
        }
        assert set(task["depends_on"]) == expected


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
                "revision": "0123456789abcdef",
                "license": "MIT",
                "dependency_closure_verified": True,
            },
            "compatibility": {"minecraft": "1.21.1", "loader": "fabric"},
        }
    ]
    plan = compile_evidence_first_plan(
        prompt,
        _design(prompt, "trade.transaction"),
        component_catalog=components,
    )

    assert plan["verified_provides"] == []
    assert [gap["missing_provides"] for gap in plan["gap_catalog"]] == [
        ["capability:trade.transaction"]
    ]
    assert any(
        "capability:trade.transaction" in task["provides"]
        for task in plan["tasks"]
    )
    binding = plan["acceptance_release_bindings"][0]
    assert binding["capability"] == "trade.transaction"
    assert binding["status"] == "planned_gap"
    assert binding["component_refs"] == []


def test_inventory_symbol_without_exact_semantic_alias_does_not_retain(tmp_path) -> None:
    prompt = "Keep the weather compass."
    capability = "item.weather_compass"
    source = tmp_path / "src" / "main" / "java" / "example" / "WeatherCompass.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example;\npublic final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    design = _design(prompt, capability)
    design["_existing_project_inventory"] = inspect_project_inventory(tmp_path).to_dict()

    plan = compile_evidence_first_plan(prompt, design)

    assert f"capability:{capability}" not in plan["verified_provides"]
    assert [gap["missing_provides"] for gap in plan["gap_catalog"]] == [
        [f"capability:{capability}"]
    ]
    assert plan["reuse_decisions"][0]["action"] == "fresh"


def test_task_graph_rejects_missing_provider_edge_even_with_valid_hashes() -> None:
    prompt = "Add a machine."
    plan = compile_evidence_first_plan(
        prompt,
        _design(prompt, "automation.machine"),
    )
    tampered = copy.deepcopy(plan)
    dependent = next(task for task in tampered["tasks"] if task["depends_on"])
    dependent["depends_on"] = []
    dependent["task_sha256"] = ""
    dependent["task_sha256"] = _hash_without(dependent, "task_sha256")
    _rehash(tampered)

    with pytest.raises(EvidencePlanError, match="deterministic template DAG"):
        validate_evidence_first_plan(tampered, prompt=prompt)


def test_design_modules_do_not_override_explicit_semantic_authority() -> None:
    prompt = "Add trade. Add quests."
    design = _design(prompt, "trade.transaction", "quest.state")
    design["modules"] = [{"plugin_id": "placeholder", "reason": "placeholder"}]

    plan = compile_evidence_first_plan(prompt, design)

    assert [
        item["capability"] for item in plan["request_catalog"]["requirements"]
    ] == ["trade.transaction", "quest.state"]
    assert {item["missing_provides"][0] for item in plan["gap_catalog"]} == {
        "capability:trade.transaction",
        "capability:quest.state",
    }


def test_long_semantic_capability_is_not_truncated_to_identifier_budget() -> None:
    capability = (
        "interdimensional_player_owned_energy_distribution_network_with_audited_access"
    )
    prompt = capability.replace("_", " ") + "."
    plan = compile_evidence_first_plan(prompt, _design(prompt, capability))

    result_cap = plan["request_catalog"]["requirements"][0]["capability"]
    assert result_cap == capability
    assert re.match(r"^[a-z0-9_.]+$", result_cap)


def test_unresolved_target_defers_semantic_implementation_planning() -> None:
    prompt = "Add quests."
    design = {
        "modules": [{"plugin_id": "quests", "reason": "quests"}],
        "_evidence_request_catalog": _request_catalog(prompt, "quest.state"),
    }

    with pytest.raises(EvidencePlanError, match="planning is deferred"):
        compile_evidence_first_plan(prompt, design)


def test_multiloader_creates_cross_module_owned_loader_binding() -> None:
    prompt = "Add quests to both loader leaves."
    plan = compile_evidence_first_plan(
        prompt,
        _design(prompt, "quest.state"),
        target_decision={
            "target": {"minecraft_version": "1.21.1", "loader": "multiloader"},
            "project_topology": {
                "module_ids": [":common", ":fabric", ":neoforge"],
                "loaders": ["fabric", "neoforge"],
            },
        },
    )
    assert plan["branch_predicates"]["needs_loader_leaf"]["status"] == "ACTIVE"
    loader_tasks = [
        task
        for task in plan["tasks"]
        if len(
            {
                anchor.get("module_id")
                for anchor in task.get("owned_anchors", [])
                if isinstance(anchor, dict) and anchor.get("module_id")
            }
        )
        > 1
    ]
    assert loader_tasks
    assert all(task["depends_on"] for task in loader_tasks)


@pytest.mark.parametrize(
    ("capability", "semantic_type", "active", "inactive"),
    [
        ("performance.optimization", "software_quality", "needs_mixin", "needs_registry"),
        ("planet.special_mineral", "gameplay_mechanic", "needs_worldgen", "needs_network"),
        ("ui.menu", "gameplay_mechanic", "needs_client_render", "needs_persistence"),
    ],
)
def test_template_feature_model_activates_only_applicable_subsystems(
    capability: str,
    semantic_type: str,
    active: str,
    inactive: str,
) -> None:
    features = requirement_branch_features(
        {"capability": capability, "semantic_type": semantic_type}
    )
    assert active in features
    assert inactive not in features


def test_pre_target_catalog_is_reused_and_stale_prompt_is_rejected() -> None:
    prompt = "Add a quest system."
    original = _request_catalog(prompt, "quest.state")
    design = _design(prompt, "quest.state")
    design["_evidence_request_catalog"] = original

    reused = compile_evidence_first_plan(prompt, design)
    assert reused["request_catalog"] == original

    with pytest.raises(EvidencePlanError, match="stale"):
        compile_evidence_first_plan(prompt + " Changed.", design)


def test_recomputed_hashes_cannot_forge_unverified_retain_coverage() -> None:
    prompt = "Add trade."
    plan = compile_evidence_first_plan(
        prompt,
        _design(prompt, "trade.transaction"),
        component_catalog=[
            {
                "component_id": "external_trade_candidate",
                "kind": "symbol",
                "locator": "Trade.java#Trade",
                "content_sha256": "sha256:" + "d" * 64,
                "provides": ["capability:trade.transaction"],
                "verification_status": "verified",
                "bound_to_project": True,
                "provenance": {
                    "origin": "external",
                    "repository": "https://example.invalid/repo",
                    "revision": "0" * 40,
                    "license": "MIT",
                    "dependency_closure_verified": True,
                },
                "compatibility": {"minecraft": "1.21.1"},
            }
        ],
    )
    forged = copy.deepcopy(plan)
    decision = forged["reuse_decisions"][0]
    decision.update(
        {
            "action": "retain",
            "component_refs": ["external_trade_candidate"],
            "source_refs": [],
            "external_receipt": {},
            "evidence_status": "verified",
        }
    )
    decision["decision_sha256"] = ""
    decision["decision_sha256"] = _hash_without(decision, "decision_sha256")
    _rehash(forged)

    with pytest.raises(EvidencePlanError, match="without exact capability evidence"):
        validate_evidence_first_plan(forged, prompt=prompt)


def test_validated_multiloader_inventory_binds_cross_module_anchors(tmp_path) -> None:
    (tmp_path / "settings.gradle").write_text(
        "rootProject.name = 'multi'\ninclude ':common', ':fabric', ':neoforge'\n",
        encoding="utf-8",
    )
    for module in ("common", "fabric", "neoforge"):
        root = tmp_path / module
        root.mkdir()
        (root / "build.gradle").write_text("plugins {}\n", encoding="utf-8")
        java = root / "src" / "main" / "java" / "example" / f"{module.title()}Root.java"
        java.parent.mkdir(parents=True)
        java.write_text(
            f"package example;\npublic final class {module.title()}Root {{}}\n",
            encoding="utf-8",
        )
    fabric_meta = tmp_path / "fabric" / "src" / "main" / "resources" / "fabric.mod.json"
    fabric_meta.parent.mkdir(parents=True)
    fabric_meta.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "multi",
                "version": "1.0.0",
                "depends": {"minecraft": "1.21.1"},
            }
        ),
        encoding="utf-8",
    )
    neo_meta = tmp_path / "neoforge" / "src" / "main" / "resources" / "META-INF" / "neoforge.mods.toml"
    neo_meta.parent.mkdir(parents=True)
    neo_meta.write_text(
        'license="MIT"\n[[mods]]\nmodId="multi"\nversion="1.0.0"\ndisplayName="Multi"\n',
        encoding="utf-8",
    )
    prompt = "Add quests."
    design = _design(prompt, "quest.state")
    design["_existing_project_inventory"] = inspect_project_inventory(tmp_path).to_dict()
    plan = compile_evidence_first_plan(prompt, design)

    assert plan["branch_predicates"]["needs_loader_leaf"]["status"] == "ACTIVE"
    cross_module_tasks = [
        task
        for task in plan["tasks"]
        if {
            anchor.get("module_id")
            for anchor in task.get("owned_anchors", [])
            if isinstance(anchor, dict) and anchor.get("module_id")
        }
        >= {"common", "fabric", "neoforge"}
    ]
    assert cross_module_tasks
    assert plan["ownership_context"]["module_id"] == "common"


def _hole_fill_response(messages) -> tuple[str, int]:
    payload = json.loads(messages[-1]["content"].split("\n", 1)[1])
    if isinstance(payload.get("modules"), list):
        holes = payload["modules"][0]["implementation_template"]["holes"]
        response = "\n".join(
            f"### Hole {index}\nDecision: Implement only the supplied host-owned contract.\n"
            "Steps:\n- Implement the bounded task-local behavior.\n"
            "Bindings: none\nReferences: none\n"
            "Verification: Run the host-owned required gates.\nUncertainties: none"
            for index, _hole in enumerate(holes, 1)
        )
        return response, len(holes)
    pages = payload["pages"]
    response = "\n".join(
        f"BEGIN PAGE {page['page_id']}\nBEGIN {hole['hole_id']}\n"
        "Decision: Implement only the supplied host-owned contract.\n"
        "Steps:\n- Implement the bounded task-local behavior.\n"
        "Bindings: none\nReferences: none\n"
        "Verification: Run the host-owned required gates.\nUncertainties: none\n"
        f"END {hole['hole_id']}\nEND PAGE {page['page_id']}"
        for page in pages
        for hole in page["holes"]
    )
    return response, sum(len(page["holes"]) for page in pages)


def test_host_task_pages_allow_only_bounded_implementation_hole_fills() -> None:
    prompt = "Add a machine with a very specific observable behavior."
    design = _design(prompt, "automation.machine")
    plan = compile_evidence_first_plan(prompt, design)
    batches = _evidence_host_batches(plan)

    class Router:
        def __init__(self) -> None:
            self.calls = 0
            self.requested_hole_counts: list[int] = []

        def generate_text(self, _role, messages, **_kwargs):
            self.calls += 1
            response, hole_count = _hole_fill_response(messages)
            self.requested_hole_counts.append(hole_count)
            return response

    router = Router()
    modules, _assets, _tests = CompleteGameDesignPlanner(router)._expand_batches(
        batches,
        prompt=prompt,
        game_design={**design, "_evidence_first_plan": plan},
        evidence_mode=True,
    )

    assert tuple(module.module_id for module in modules) == tuple(
        task["task_id"] for task in plan["tasks"]
    )
    by_id = {module.module_id: module for module in modules}
    for task in plan["tasks"]:
        module = by_id[task["task_id"]]
        assert module.depends_on == tuple(task["depends_on"])
        assert module.config["evidence_task"]["task_id"] == task["task_id"]
        template = module.config["implementation_template"]
        allowed = set(template["completion_policy"]["required_hole_ids"])
        filled = {
            item["hole_id"] for item in module.config["model_fill"]["hole_fills"]
        }
        assert filled == allowed
    assert router.calls == len(router.requested_hole_counts)
    assert router.calls > 0
    assert all(0 < count <= 12 for count in router.requested_hole_counts)
