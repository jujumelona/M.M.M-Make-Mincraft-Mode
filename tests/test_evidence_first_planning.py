from __future__ import annotations

import copy
import json

import pytest

from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner, _evidence_host_batches
from minecraft_mod_ai.evidence_first_planning import (
    EvidencePlanError,
    _hash_without,
    build_request_catalog,
    compile_evidence_first_plan,
    validate_evidence_first_plan,
)
from minecraft_mod_ai.minecraft_template_catalog import requirement_branch_features
from minecraft_mod_ai.project_inventory import inspect_project_inventory


def _design(*capabilities: str) -> dict[str, object]:
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
    first = compile_evidence_first_plan(prompt, _design("machine"))
    second = compile_evidence_first_plan(prompt, _design("machine"))

    assert first == second
    assert first["plan_sha256"].startswith("sha256:")
    assert len(first["request_catalog"]["requirements"]) == 3
    assert len(first["tasks"]) > 10


def test_prompt_only_catalog_preserves_each_authored_requirement_clause() -> None:
    prompt = "Add a machine with saved state, synced packets, and a screen."
    catalog = build_request_catalog(prompt, {})

    assert [item["capability"] for item in catalog["requirements"]] == [
        "automation.machine",
        "network.action_sync",
        "ui.menu",
    ]
    assert [item["source_span"]["text"] for item in catalog["requirements"]] == [
        "Add a machine with saved state",
        "synced packets",
        "and a screen.",
    ]


def test_design_module_name_cannot_add_semantic_authority() -> None:
    catalog = build_request_catalog("Add quests.", _design("placeholder"))

    assert [item["capability"] for item in catalog["requirements"]] == ["quest.state"]
    assert all(item["provenance_role"] == "explicit" for item in catalog["requirements"])


def test_frozen_catalog_is_reused_without_design_module_augmentation() -> None:
    prompt = "Add trade. Add quests."
    catalog = build_request_catalog(prompt, _design("trade"))
    catalog["requirements"] = [
        item for item in catalog["requirements"] if item["capability"] != "quest.state"
    ]
    catalog["catalog_sha256"] = _hash_without(catalog, "catalog_sha256")
    design = _design("placeholder")
    design["_evidence_request_catalog"] = catalog

    reused = build_request_catalog(prompt, design)

    assert [item["capability"] for item in reused["requirements"]] == [
        "trade.transaction"
    ]
    assert reused["catalog_sha256"] == _hash_without(reused, "catalog_sha256")


def test_machine_vertical_dag_uses_exact_provider_edges() -> None:
    prompt = "Add a machine with saved state, synced packets, and a screen."
    plan = compile_evidence_first_plan(prompt, _design("machine"))

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
        "Add trade.",
        _design("trade"),
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
    source = tmp_path / "src" / "main" / "java" / "example" / "WeatherCompass.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example;\npublic final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    design = _design("weather_compass")
    design["_existing_project_inventory"] = inspect_project_inventory(tmp_path).to_dict()

    plan = compile_evidence_first_plan("Keep the weather compass.", design)

    assert "capability:weather_compass" in plan["verified_provides"]
    missing = {
        provided
        for gap in plan["gap_catalog"]
        for provided in gap["missing_provides"]
    }
    assert missing.isdisjoint(plan["verified_provides"])
    assert len(plan["gap_catalog"]) == 1
    assert plan["tasks"]
    binding = plan["acceptance_release_bindings"][0]
    assert binding["status"] == "planned_gap"
    assert binding["component_refs"] == []


def test_task_graph_rejects_missing_provider_edge_even_with_valid_hashes() -> None:
    plan = compile_evidence_first_plan("Add a block.", _design("block"))
    tampered = copy.deepcopy(plan)
    dependent = next(task for task in tampered["tasks"] if task["depends_on"])
    dependent["depends_on"] = []
    dependent["task_sha256"] = ""
    dependent["task_sha256"] = _hash_without(dependent, "task_sha256")
    _rehash(tampered)

    with pytest.raises(EvidencePlanError, match="deterministic template DAG"):
        validate_evidence_first_plan(tampered, prompt="Add a block.")


def test_design_modules_do_not_override_prompt_semantic_capabilities() -> None:
    plan = compile_evidence_first_plan(
        "Add trade. Add quests.",
        _design("placeholder"),
    )

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
    plan = compile_evidence_first_plan(prompt, _design())

    result_cap = plan["request_catalog"]["requirements"][0]["capability"]
    assert result_cap
    import re

    assert re.match(r"^[a-z0-9_.]+$", result_cap)


def test_unresolved_target_defers_semantic_implementation_planning() -> None:
    with pytest.raises(EvidencePlanError, match="planning is deferred"):
        compile_evidence_first_plan(
            "Add quests.",
            {"modules": [{"plugin_id": "quests", "reason": "quests"}]},
        )


def test_multiloader_creates_cross_module_owned_loader_binding() -> None:
    plan = compile_evidence_first_plan(
        "Add quests to both loader leaves.",
        _design("quests"),
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
    original = compile_evidence_first_plan(prompt, _design("quests"))["request_catalog"]
    design = _design("quests")
    design["_evidence_request_catalog"] = original
    reused = compile_evidence_first_plan(prompt, design)
    assert reused["request_catalog"] == original

    with pytest.raises(EvidencePlanError, match="stale"):
        compile_evidence_first_plan(prompt + " Changed.", design)


def test_recomputed_hashes_cannot_forge_unverified_retain_coverage() -> None:
    plan = compile_evidence_first_plan(
        "Add trade.",
        _design("trade"),
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
        validate_evidence_first_plan(forged, prompt="Add trade.")


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
    design = _design("quests")
    design["_existing_project_inventory"] = inspect_project_inventory(tmp_path).to_dict()
    plan = compile_evidence_first_plan("Add quests.", design)

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
    prompt = "Add a block with a very specific observable behavior."
    plan = compile_evidence_first_plan(prompt, _design("block"))
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
        game_design={**_design("block"), "_evidence_first_plan": plan},
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
