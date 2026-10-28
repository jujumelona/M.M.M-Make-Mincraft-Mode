from __future__ import annotations

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.content_design_graph import compile_content_graph


def test_zero_entity_feature_remains_executable_work(monkeypatch):
    monkeypatch.setattr("minecraft_mod_ai.content_design_graph.run_record_template",
                        lambda *args, **kwargs: {"records": []})
    row = {"requirement_id": "request_planet", "statement": "Travel to planets",
           "design_context": {"source_section": "request_scope", "source_concern": "features",
                              "source_feature": {"implementation_role": "gameplay"}}}
    result = compile_content_graph("Travel to planets", object(),
                                   request_catalog={"requirements": [row], "host_constraints": {}})
    assert result["modules"] == []
    assert result["_request_requirements"][0]["requirement_id"] == "request_planet"
    assert result["_residual_request_features"][0]["statement"] == "Travel to planets"


def test_feature_glue_uses_registered_java_and_real_fabric_startup():
    from minecraft_mod_ai.artifact_job import (
        canonical_main_entrypoints,
        validate_artifact_job_graph,
    )
    from minecraft_mod_ai.authored_feature_implementation import (
        compile_request_feature_jobs,
    )
    from minecraft_mod_ai.host_version_catalog import host_target

    adapter = host_target("26.1.2")
    plan = AuthoredPlan("Travel to planets", "Saved worksheet", request_scope={
        "requirements": [{
            "requirement_id": "request_planet", "statement": "Travel to planets",
            "search_query": "space travel", "implementation_role": "gameplay",
        }],
        "behavior_requirements": [{
            "behavior_id": "behavior_travel", "requirement_id": "request_planet",
            "intent": "Travel to planets", "source_quote": "Travel to planets",
            "trigger": "Requested journey", "effects": "Arrive at a planet",
            "state_reads": [], "state_writes": [],
        }],
    })
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=adapter, mod_id="space", package_name="test.space",
        content_modules=(), content_jobs=(),
    )
    assert modules[0].kind == "integration"
    assert modules[0].config["requirement_refs"] == ["request_planet"]
    assert jobs[0].canonical_leaf == "minecraft/event/callback"
    validate_artifact_job_graph(jobs, module_ids=[m.module_id for m in modules])
    entrypoints = canonical_main_entrypoints(jobs)
    spec = jobs[0].deterministic_inputs["_canonical_inputs"]["event_callback_input"]
    assert entrypoints == (spec["bindings"]["package_name"] + "." + spec["bindings"]["class_name"],)
    assert "onInitialize" in spec["output_schema"]["pattern"]
    assert "Travel to planets" in spec["requirement"]
    assert "player_join" not in spec["bindings"]
    from minecraft_mod_ai.integrity_bootstrap import get_integrity_authority
    get_integrity_authority().types.validate_input(
        "minecraft/event/callback:input", jobs[0].deterministic_inputs["_canonical_inputs"],
    )
    import json

    from minecraft_mod_ai.canonical_generators import generate_canonical_leaf

    class Router:
        def generate_text(self, role, messages, **kwargs):
            assert role == "coder"
            assert "Travel to planets" in messages[-1]["content"]
            return json.dumps(
                f"package {spec['bindings']['package_name']}; "
                f"public final class {spec['bindings']['class_name']} implements net.fabricmc.api.ModInitializer "
                "{ public void onInitialize() { System.out.println(\"test startup\"); } }"
            )

    generated = generate_canonical_leaf(jobs[0].deterministic_inputs["_canonical_inputs"],
                                        leaf_id=jobs[0].canonical_leaf, router=Router(),
                                        authority=get_integrity_authority())
    assert "onInitialize" in generated["event_callback_artifact"]
    assert generated["event_callback_receipt"]["context_id"] == adapter.version_context.context_id


def test_explicit_gui_feature_uses_existing_screen_without_extra_server_class():
    from types import SimpleNamespace

    from minecraft_mod_ai.authored_feature_implementation import (
        compile_request_feature_jobs,
    )

    plan = AuthoredPlan("Shipyard screen", "Saved", request_scope={"requirements": [{
        "requirement_id": "screen", "statement": "Show shipyard screen", "implementation_role": "interface",
    }]})
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=SimpleNamespace(), mod_id="space", package_name="test.space",
        content_modules=(SimpleNamespace(kind="gui", config={"requirement_refs": ["screen"]}),),
        content_jobs=(),
    )
    assert modules == jobs == ()


def test_non_executable_context_heading_does_not_spawn_fake_mod_initializer():
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.host_version_catalog import host_target

    plan = AuthoredPlan("Space mod with combat", "Saved", request_scope={
        "requirements": [
            {"requirement_id": "scope", "statement": "Space mod", "implementation_role": "gameplay"},
            {"requirement_id": "combat", "statement": "Fight a hostile creature",
             "implementation_role": "gameplay"},
        ],
        "behavior_requirements": [
            {"behavior_id": "scope_behavior", "requirement_id": "scope",
             "source_status": "source_preserved_unresolved",
             "intent": "Space mod", "source_quote": "Space mod", "effects": "Space mod",
             "trigger": "", "state_reads": [], "state_writes": []},
            {"behavior_id": "fight_behavior", "requirement_id": "combat",
             "source_status": "grounded", "intent": "Fight a hostile creature",
             "source_quote": "Fight a hostile creature",
             "trigger": "Player attacks a hostile creature",
             "effects": "The creature takes damage", "state_reads": [], "state_writes": []},
        ],
    })
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=host_target("26.1.2"), mod_id="space", package_name="test.space",
        content_modules=(), content_jobs=(),
    )
    assert len(modules) == len(jobs) == 1
    assert modules[0].config["requirement_refs"] == ["combat"]
    assert all("scope" not in module.module_id for module in modules)
    # The context heading remains in request_scope; it is not a claimed
    # implemented callback or a disguised gameplay success.
    assert len(plan.request_scope["requirements"]) == 2


def test_only_unresolved_context_is_not_misreported_as_executable():
    import pytest

    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    plan = AuthoredPlan("Space mod", "Saved", request_scope={
        "requirements": [{"requirement_id": "scope", "statement": "Space mod",
                          "implementation_role": "gameplay"}],
        "behavior_requirements": [
            {"behavior_id": "scope_behavior", "requirement_id": "scope",
             "source_status": "source_preserved_unresolved",
             "intent": "Space mod", "effects": "Space mod", "trigger": "",
             "state_reads": [], "state_writes": []},
        ],
    })
    with pytest.raises(ValueError, match="REQUEST_FEATURE_EXECUTABLE_BEHAVIOR_REQUIRED"):
        compile_request_feature_jobs(
            plan, adapter=object(), mod_id="space", package_name="test.space",
            content_modules=(), content_jobs=(),
        )


def test_feature_integration_receives_content_source_even_without_matching_requirement_refs():
    """Log 34: shipyard GUI existed, but request-specific feature saw no site."""
    import json
    from types import SimpleNamespace
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.host_version_catalog import host_target

    plan = AuthoredPlan("Modular spacecraft construction", "Saved", request_scope={
        "requirements": [{
            "requirement_id": "ship_builder",
            "statement": "Place ship sections into a spacecraft",
            "implementation_role": "gameplay",
        }],
        "behavior_requirements": [{
            "behavior_id": "build_section", "requirement_id": "ship_builder",
            "intent": "Build modular spacecraft", "source_quote": "Place ship sections",
            "trigger": "Player confirms ship section selection",
            "effects": "Builds selected section", "state_reads": [],
            "state_writes": [],
        }],
    })
    # Content authored before feature decomposition may preserve an unrelated
    # content entity ID and omit the request requirement_id association.
    gui = ProductionModule("shipyard_ui", "gui", {
        "requirement_refs": ["catalog_generated_shipyard"],
    })
    item = ProductionModule("spacecraft_parts", "item", {
        "requirement_refs": [],
    })
    gui.validate()
    item.validate()
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=host_target("26.1.2"), mod_id="space",
        package_name="test.space",
        content_modules=(gui, item),
        content_jobs=(
            SimpleNamespace(
                owner_module=gui.module_id,
                target_path="src/main/java/test/space/ShipyardScreen.java",
                deterministic_inputs={}, canonical_leaf="",
                produces=("shipyard_ui.source_artifact",),
            ),
            SimpleNamespace(
                owner_module=item.module_id,
                target_path="src/main/java/test/space/SpacecraftPart.java",
                deterministic_inputs={}, canonical_leaf="",
                produces=("spacecraft_parts.source_artifact",),
            ),
        ),
    )
    assert set(modules[0].depends_on) == {"shipyard_ui", "spacecraft_parts"}
    # The actual executor uses job.requires; module.depends_on is insufficient.
    assert set(jobs[0].requires) == {
        "shipyard_ui.source_artifact", "spacecraft_parts.source_artifact",
    }
    spec = jobs[0].deterministic_inputs["_canonical_inputs"]["event_callback_input"]
    paths = json.loads(spec["bindings"]["content_dependency_paths_json"])
    required = json.loads(spec["bindings"]["required_content_source_paths_json"])
    assert "src/main/java/test/space/ShipyardScreen.java" in paths
    assert "src/main/java/test/space/SpacecraftPart.java" in paths
    assert set(required) == {
        "src/main/java/test/space/ShipyardScreen.java",
        "src/main/java/test/space/SpacecraftPart.java",
    }


def test_integration_source_dependencies_are_not_inferred_from_intent_text():
    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.host_version_catalog import host_target

    plan = AuthoredPlan("Explore other dimensions", "Saved", request_scope={
        "requirements": [{
            "requirement_id": "explore",
            "statement": "Explore another dimension",
            "implementation_role": "gameplay",
        }],
        "behavior_requirements": [{
            "behavior_id": "travel", "requirement_id": "explore",
            "intent": "Travel", "source_quote": "Explore another dimension",
            "trigger": "Player enters the portal",
            "effects": "Player travels", "state_reads": [], "state_writes": [],
        }],
    })
    content = ProductionModule("unrelated_named_owner", "block", {
        "requirement_refs": ["different_requirement"],
    })
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=host_target("26.1.2"), mod_id="mods",
        package_name="test.mod", content_modules=(content,), content_jobs=(),
    )
    assert "unrelated_named_owner" in modules[0].depends_on
    assert len(jobs) == 1


def test_canonical_python_generator_java_path_is_required_before_callback_author():
    import json
    from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.host_version_catalog import host_target
    from minecraft_mod_ai.prompt_fact_types import FactType, PromptFact
    from minecraft_mod_ai.task_template_catalog import canonical_specification_input

    adapter = host_target("auto")
    jobs = expand_facts_to_jobs(
        [PromptFact(fact_id="entity_alien", fact_type=FactType.ENTITY_EXISTS,
                    subject="alien_hostile_entity")],
        mod_id="space", package_name="example.space",
        version_context=adapter.version_context,
    )
    entity = next(j for j in jobs if j.canonical_leaf == "minecraft/entity/type")
    assert not entity.target_path  # The real path is in canonical input.
    leaf_input = entity.deterministic_inputs["_canonical_inputs"]
    expected = canonical_specification_input(
        entity.canonical_leaf, leaf_input,
    )["target_path"]
    plan = AuthoredPlan("Fight alien", "Saved", request_scope={
        "requirements": [{"requirement_id": "battle", "statement": "Fight an alien",
                          "implementation_role": "gameplay"}],
        "behavior_requirements": [{
            "behavior_id": "fight_alien", "requirement_id": "battle",
            "intent": "Fight alien", "source_quote": "Fight an alien",
            "trigger": "Player attacks alien", "effects": "Alien loses health",
            "state_reads": [], "state_writes": [],
        }],
    })
    modules, produced = compile_request_feature_jobs(
        plan, adapter=adapter, mod_id="space", package_name="example.space",
        content_modules=(ProductionModule("alien_hostile_entity", "entity",
                                          {"requirement_refs": []}),),
        content_jobs=(entity,),
    )
    spec = produced[0].deterministic_inputs["_canonical_inputs"]["event_callback_input"]
    required = json.loads(spec["bindings"]["required_content_source_paths_json"])
    assert expected in required
    assert set(produced[0].requires) == set(entity.produces)


def _multi_behavior_feature_plan(*, stateful=False):
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    effects = ("A player plants a seed", "A second player takes a reward")
    rows = []
    for index, effect in enumerate(effects):
        row = {
            "behavior_id": "behavior_" + str(index),
            "requirement_id": "farming",
            "source_status": "grounded",
            "intent": effect,
            "source_quote": effect,
            "trigger": "Player interacts with farming block " + str(index),
            "effects": effect,
            "state_reads": ["owned_resource"] if stateful else [],
            "state_writes": [],
        }
        rows.append(row)
    return AuthoredPlan("Two separate farm actions", "Saved", request_scope={
        "requirements": [{
            "requirement_id": "farming", "statement": "Two separate farm actions",
            "implementation_role": "gameplay",
        }],
        "behavior_requirements": rows,
    })


def test_independent_actions_use_separate_java_compilation_units():
    import json
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.host_version_catalog import host_target
    plan = _multi_behavior_feature_plan()
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=host_target("26.1.2"), mod_id="farm", package_name="example.farm",
        content_modules=(), content_jobs=(),
    )
    assert len(modules) == len(jobs) == 2
    assert modules[0].module_id != modules[1].module_id
    assert {module.config["implementation_obligations"][0] for module in modules} == {
        "A player plants a seed", "A second player takes a reward",
    }
    rows_by_job = [
        json.loads(job.deterministic_inputs["_canonical_inputs"]["event_callback_input"]
                   ["bindings"]["construction_behaviors_json"])
        for job in jobs
    ]
    assert len({row[0]["behavior_id"] for row in rows_by_job}) == 2
    assert all(len(rows) == 1 for rows in rows_by_job)


def test_atomic_stateful_actions_stay_in_single_java_owner():
    import json
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.host_version_catalog import host_target
    plan = _multi_behavior_feature_plan(stateful=True)
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=host_target("26.1.2"), mod_id="farm", package_name="example.farm",
        content_modules=(), content_jobs=(),
    )
    assert len(modules) == len(jobs) == 1
    rows = json.loads(jobs[0].deterministic_inputs["_canonical_inputs"]
                      ["event_callback_input"]["bindings"]["construction_behaviors_json"])
    assert len(rows) == 2
