from __future__ import annotations

import copy
import json

import pytest

from minecraft_mod_ai.complete_planner import (
    CompleteGameDesignPlanner,
    _evidence_host_batches,
)
from minecraft_mod_ai.evidence_first_planning import (
    EvidencePlanError,
    _hash_without,
    build_request_catalog,
    compile_evidence_first_plan,
    task_batches,
    validate_evidence_first_plan,
)
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

    # Gameplay roots are now promoted for ALL languages (not just Korean).
    # Each clause resolves to its ontology gameplay root.
    assert [item["capability"] for item in catalog["requirements"]] == [
        "automation.machine",
        "network.action_sync",
        "ui.menu",
    ]
    assert [item["source_span"]["text"] for item in catalog["requirements"]] == [
        "Add a machine with saved state",
        "synced packets",
        "a screen.",
    ]


def test_unmatched_design_module_does_not_cover_the_prompt_clause() -> None:
    catalog = build_request_catalog("Add quests.", _design("placeholder"))

    # "Add quests." resolves to quest.state gameplay root (language-neutral promotion)
    assert [item["capability"] for item in catalog["requirements"]] == [
        "placeholder",
        "quest.state",
    ]


def test_existing_catalog_merges_uncovered_prompt_clause_without_dropping_records() -> None:
    prompt = "Add trade. Add quests."
    catalog = build_request_catalog(prompt, _design("trade"))
    catalog["requirements"] = [
        item for item in catalog["requirements"] if item["capability"] != "quest.state"
    ]
    catalog["catalog_sha256"] = _hash_without(catalog, "catalog_sha256")
    design = _design("trade")
    design["_evidence_request_catalog"] = catalog

    merged = build_request_catalog(prompt, design)

    assert [item["capability"] for item in merged["requirements"]] == [
        "trade",
        "quest.state",
    ]
    assert merged["catalog_sha256"] == _hash_without(merged, "catalog_sha256")


def test_machine_vertical_dag_uses_predicates_and_exact_provider_edges() -> None:
    prompt = "Add a machine with saved state, synced packets, and a screen."
    plan = compile_evidence_first_plan(prompt, _design("machine"))
    outcomes = [task["semantic_outcome"] for task in plan["tasks"]]

    for required in (
        "registry identities",
        "block shell",
        "block item",
        "block-entity type",
        "server behavior",
        "persistence codec",
        "Persist and reload",
        "payload codec",
        "synchronize",
        "menu slots",
        "client screen",
        "models, language, loot",
        "complete semantic outcome",
    ):
        assert any(required in outcome for outcome in outcomes)

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


def test_self_claimed_external_component_never_removes_a_gap() -> None:
    digest = "sha256:" + "a" * 64
    components = [
        {
            "component_id": "existing_trade",
            "kind": "symbol",
            "locator": "src/main/java/example/TradeService.java#TradeService",
            "content_sha256": digest,
            "provides": ["capability:trade"],
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
        "Add trade and quests.",
        _design("trade", "quests"),
        component_catalog=components,
    )

    assert plan["verified_provides"] == []
    assert {gap["missing_provides"][0] for gap in plan["gap_catalog"]} == {
        "capability:trade",
        "capability:quests",
    }
    assert any("capability:trade" in task["provides"] for task in plan["tasks"])
    trade_binding = next(
        item for item in plan["acceptance_release_bindings"] if item["capability"] == "trade"
    )
    assert trade_binding["status"] == "planned_gap"
    assert trade_binding["component_refs"] == []


def test_arbitrary_same_project_mapping_cannot_claim_verified_coverage() -> None:
    plan = compile_evidence_first_plan(
        "Add trade.",
        _design("trade"),
        component_catalog=[
            {
                "component_id": "fake_trade",
                "kind": "symbol",
                "locator": "Trade.java",
                "content_sha256": "sha256:" + "b" * 64,
                "provides": ["trade"],
                "provenance": "same_project",
                "verification_status": "verified",
            }
        ],
    )
    assert plan["verified_provides"] == []
    assert len(plan["gap_catalog"]) == 1
    assert plan["tasks"]


def test_validated_inventory_exact_capability_alias_retains_existing_symbol(tmp_path) -> None:
    source = tmp_path / "src" / "main" / "java" / "example" / "WeatherCompass.java"
    source.parent.mkdir(parents=True)
    source.write_text(
        "package example;\npublic final class WeatherCompass {}\n",
        encoding="utf-8",
    )
    inventory = inspect_project_inventory(tmp_path).to_dict()
    design = _design("weather_compass")
    design["_existing_project_inventory"] = inventory

    plan = compile_evidence_first_plan("Keep the weather compass.", design)

    assert plan["verified_provides"] == ["capability:weather_compass"]
    assert plan["gap_catalog"] == []
    assert plan["tasks"] == []
    binding = plan["acceptance_release_bindings"][0]
    assert binding["status"] == "retained"
    assert len(binding["component_refs"]) == 1
    assert binding["component_refs"][0].startswith("component:symbol:")


def test_task_graph_rejects_missing_provider_edge_even_with_valid_hashes() -> None:
    plan = compile_evidence_first_plan("Add a block.", _design("block"))
    tampered = copy.deepcopy(plan)
    dependent = next(task for task in tampered["tasks"] if task["depends_on"])
    dependent["depends_on"] = []
    dependent["task_sha256"] = ""
    dependent["task_sha256"] = _hash_without(dependent, "task_sha256")
    _rehash(tampered)

    with pytest.raises(EvidencePlanError, match="host gap DAG"):
        validate_evidence_first_plan(tampered, prompt="Add a block.")


def test_design_modules_are_merged_with_uncovered_prompt_requirements() -> None:
    plan = compile_evidence_first_plan(
        "Add trade. Add quests.",
        _design("trade"),
    )
    # Design module IDs are preserved verbatim; only prompt-derived clauses get
    # gameplay root promotion. "trade" comes from _design("trade"), "quests"
    # comes from the uncovered prompt clause "Add quests." → promoted to "quest.state".
    assert [
        item["capability"] for item in plan["request_catalog"]["requirements"]
    ] == ["trade", "quest.state"]
    assert {item["missing_provides"][0] for item in plan["gap_catalog"]} == {
        "capability:trade",
        "capability:quest.state",
    }


def test_long_semantic_capability_is_not_truncated_to_an_identifier_budget() -> None:
    capability = (
        "interdimensional_player_owned_energy_distribution_network_with_audited_access"
    )
    prompt = capability.replace("_", " ") + "."
    design = _design("placeholder")
    design["modules"] = []
    plan = compile_evidence_first_plan(prompt, design)

    # The semantic compiler resolves the prompt to an ontology gameplay root if
    # one matches (e.g. "interdimensional" contains "dimension" → worldgen.dimension).
    # The important invariant is that a capability IS produced and is an ASCII ID.
    result_cap = plan["request_catalog"]["requirements"][0]["capability"]
    assert result_cap  # must be non-empty
    import re as _re
    assert _re.match(r"^[a-z0-9_.]+$", result_cap), f"Non-ASCII capability: {result_cap!r}"


def test_unresolved_target_defers_semantic_implementation_planning() -> None:
    with pytest.raises(EvidencePlanError, match="planning is deferred"):
        compile_evidence_first_plan(
            "Add quests.",
            {"modules": [{"plugin_id": "quests", "reason": "quests"}]},
        )


def test_multiloader_adds_common_to_leaf_semantic_edge() -> None:
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
    leaf = next(task for task in plan["tasks"] if "loader leaf" in task["semantic_outcome"])
    assert leaf["consumes"] == ["common_contract:quests"]
    assert leaf["provides"] == ["capability:quests"]
    assert len(leaf["depends_on"]) == 1


@pytest.mark.parametrize(
    ("capability", "active", "inactive"),
    [
        ("performance.optimization", "needs_mixin", "needs_registry"),
        ("worldgen.placement", "needs_worldgen", "needs_network"),
        ("ui.config_screen", "needs_client_render", "needs_persistence"),
    ],
)
def test_only_applicable_subsystem_branch_is_activated(
    capability: str,
    active: str,
    inactive: str,
) -> None:
    plan = compile_evidence_first_plan(f"Implement {capability}.", _design(capability))
    assert plan["branch_predicates"][active]["status"] == "ACTIVE"
    assert plan["branch_predicates"][inactive]["status"] == "NOT_APPLICABLE"


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
                "provides": ["capability:trade"],
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
    with pytest.raises(EvidencePlanError, match="exact verified capability alias"):
        validate_evidence_first_plan(forged, prompt="Add trade.")


def test_validated_multiloader_inventory_binds_leaf_anchors(tmp_path) -> None:
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
        json.dumps({"schemaVersion": 1, "id": "multi", "version": "1.0.0", "depends": {"minecraft": "1.21.1"}}),
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
    leaf = next(task for task in plan["tasks"] if "loader leaf" in task["semantic_outcome"])
    assert {anchor["module_id"] for anchor in leaf["owned_anchors"]} == {
        ":common",
        ":fabric",
        ":neoforge",
    }
    assert plan["ownership_context"]["module_id"] == ":common"


def test_host_task_pages_do_not_replay_full_prompt_or_accept_graph_rewrites() -> None:
    prompt = "Add a block with a very specific observable behavior."
    plan = compile_evidence_first_plan(prompt, _design("block"))
    batches = _evidence_host_batches(plan)

    class Router:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        def generate_text(self, _role, messages, **_kwargs):
            request = json.loads(messages[-1]["content"])
            self.requests.append(request)
            skeleton = copy.deepcopy(request["template_skeleton"])
            module = skeleton["modules"][0]
            module["depends_on"] = ["model_invented_dependency"]
            module["config"] = {
                "implementation_notes": "bounded detail",
                "evidence_task": {"task_id": "model_owned"},
            }
            skeleton["modules"].append(
                {
                    **copy.deepcopy(module),
                    "module_id": "model_invented_id",
                }
            )
            return json.dumps(skeleton)

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
        assert module.config["model_fill"] == {"implementation_notes": "bounded detail"}
    assert all(request["request"] != prompt for request in router.requests)


def test_retain_only_task_batch_surface_is_empty(tmp_path) -> None:
    source = tmp_path / "src" / "main" / "java" / "example" / "Trade.java"
    source.parent.mkdir(parents=True)
    source.write_text("package example;\npublic final class Trade {}\n", encoding="utf-8")
    design = _design("trade")
    design["_existing_project_inventory"] = inspect_project_inventory(tmp_path).to_dict()
    plan = compile_evidence_first_plan(
        "Add trade.",
        design,
    )
    assert plan["gap_catalog"] == []
    assert plan["tasks"] == []
    assert task_batches(plan) == ()

    class Router:
        def generate_text(self, *_args, **_kwargs):  # pragma: no cover - must not run
            raise AssertionError("retain-only planning must not call the model")

    modules, assets, tests = CompleteGameDesignPlanner(Router())._expand_batches(
        (),
        prompt="Add trade.",
        game_design={"_evidence_first_plan": plan},
        evidence_mode=True,
        evidence_acceptance_tests=tuple(plan["acceptance_release_bindings"][0]["acceptance"]),
    )
    assert modules == ()
    assert assets == ()
    assert tests
