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
