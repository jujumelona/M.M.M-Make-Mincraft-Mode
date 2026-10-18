from __future__ import annotations

import json

import pytest

from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS


def sections():
    result = {}
    for name in ("integration", "algorithm"):
        result[name] = {"specification": {key: [] for key in DETAIL_RECORDS[name]},
                        "constraint_evidence_refs": []}
        result[name]["specification"]["inapplicable_concerns"] = []
    result["integration"]["specification"]["entry_points"] = [
        {"boundary": "server", "owner": "server", "trigger": "launch_sequence_initiated"}]
    result["algorithm"]["specification"]["steps"] = [
        {key: "Travel to a planet" for key in DETAIL_RECORDS["algorithm"]["steps"].split()}]
    return result


def scope():
    return {"requirements": [{"requirement_id": "planet", "statement": "Travel to planets",
                               "implementation_role": "gameplay", "search_query": "space travel"}]}


def _scope_with_accepted_construction_behavior():
    """Production consumes an accepted behavior ledger, not an invented fallback."""
    from minecraft_mod_ai.construction_task import BehaviorRequirement

    ledger = scope()
    ledger["behavior_requirements"] = [BehaviorRequirement(
        "behavior_planet", "planet", "Travel to planets",
        "Travel to planets", "launch_sequence_initiated",
        "Move to the chosen planet",
    ).to_dict()]
    return ledger


def test_custom_trigger_is_owned_before_typed_authoring_without_fake_event():
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
    from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues

    design = sections()
    owners = author_runtime_ownership(None, scope(), design)
    assert owners[0]["owner_module"] == "feature_planet"
    assert "integration.entry_points[0]" in owners[0]["worksheet_record_refs"]
    plan = author_typed_plan_ir(None, "Planet travel", design, {}, runtime_owners=owners)
    assert plan["event_bindings"] == []
    assert not any(fn["id"] == "logic_dispatch" for fn in plan["functions"])
    assert typed_plan_support_issues(design, plan, runtime_owners=owners) == ()
    assert typed_plan_support_issues(design, plan)


def test_independent_ownership_calls_use_measured_model_slots(monkeypatch):
    import threading

    from minecraft_mod_ai.planner_budget import PlannerBudget
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership

    monkeypatch.setattr("minecraft_mod_ai.parallel_model_tasks.router_native_model_parallelism",
                        lambda *args, **kwargs: 2)
    barrier = threading.Barrier(2, timeout=3)

    class Router:
        def generate_text(self, role, messages, **kwargs):
            record = json.loads(messages[1]["content"])["record_ref"]
            barrier.wait()
            return json.dumps("planet" if record.startswith("integration.") else "crew")

    ledger = scope()
    ledger["requirements"].append({"requirement_id": "crew", "statement": "Recruit a crew",
                                    "implementation_role": "gameplay", "search_query": "crew recruitment"})
    budget = PlannerBudget()
    owners = author_runtime_ownership(Router(), ledger, sections(), budget=budget)
    assert [row["feature_id"] for row in owners] == ["crew", "planet"]
    assert budget.call_count == 2
    assert owners[1]["worksheet_record_refs"] == ["integration.entry_points[0]"]


def test_owner_api_context_contains_actual_signatures_and_preserves_approved_inputs(tmp_path):
    from minecraft_mod_ai.request_runtime_ownership import (
        generated_owner_source_context,
    )

    source = tmp_path / "src/main/java/test/space/PlanetRegistry.java"
    source.parent.mkdir(parents=True)
    source.write_text("package test.space; public class PlanetRegistry {\n"
                      "public static final int DISTANCE = 42;\n"
                      + "// registry details\n" * 200 +
                      "public static String destinationForShip(long shipId) { return \"planet\"; }\n}")
    paths = []
    for name in ("AuthoredProgram", "AuthoredStateModel"):
        path = source.parent / (name + ".java")
        path.write_text("package test.space; public class " + name + " {\n" +
                        "\n".join(f"public static final int VALUE_{n} = {n};" for n in range(300)) + "\n}")
        paths.append(str(path.relative_to(tmp_path)))
    paths.append(str(source.relative_to(tmp_path)))
    spec = {"bindings": {"content_dependency_paths_json": json.dumps(paths),
                         "class_name": "PlanetFeature", "package_name": "test.space"},
            "requirement": "Travel to a planet"}
    before = json.dumps(spec, sort_keys=True)
    result = generated_owner_source_context(tmp_path, spec)
    assert "destinationForShip(long shipId)" in json.dumps(result)
    assert result[0]["sha256"]
    assert json.dumps(spec, sort_keys=True) == before


def test_runtime_ownership_cannot_hide_a_nonexistent_record():
    import pytest

    from minecraft_mod_ai.request_runtime_ownership import owned_record_refs

    with pytest.raises(ValueError, match="RUNTIME_OWNER_RECORD_UNKNOWN"):
        owned_record_refs(sections(), [{"feature_id": "planet", "owner_module": "feature_planet",
                                      "worksheet_record_refs": ["algorithm.steps[99]"]}])


def test_moving_one_record_does_not_claim_coverage_of_its_unmoved_peer():
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
    from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues

    original = sections()
    all_owners = [{"feature_id": "planet", "owner_module": "feature_planet",
                   "worksheet_record_refs": ["algorithm.steps[0]", "integration.entry_points[0]"]}]
    plan = author_typed_plan_ir(None, "Planet travel", original, {}, runtime_owners=all_owners)
    original["algorithm"]["specification"]["steps"].append(
        {key: "Unmoved second operation" for key in DETAIL_RECORDS["algorithm"]["steps"].split()})
    assert "algorithm.steps" in typed_plan_support_issues(original, plan, runtime_owners=all_owners)


@pytest.mark.parametrize("peer_owned", [True, False])
def test_ui_owned_atomic_record_retains_exact_action_after_projection(monkeypatch, peer_owned):
    from minecraft_mod_ai.request_runtime_ownership import (
        runtime_sections,
        ui_record_action_guards,
    )
    from minecraft_mod_ai.typed_plan_authoring import author_semantic_game_dispatch

    original = sections()
    original["algorithm"]["specification"]["steps"].append(
        {key: "Buy a ship" for key in DETAIL_RECORDS["algorithm"]["steps"].split()})
    owners = [
        {"feature_id": "planet", "owner_module": "feature_planet",
         "worksheet_record_refs": (["algorithm.steps[0]"] if peer_owned else []) + ["integration.entry_points[0]"]},
        {"feature_id": "__typed_ui__", "owner_module": "authored_typed_plan", "ui_action": "ship_purchase",
         "worksheet_record_refs": ["algorithm.steps[1]"]},
    ]
    guards = ui_record_action_guards(original, owners)
    assert guards == {"algorithm.steps[0]": "ship_purchase"}
    projected = runtime_sections(original, owners, for_ui=True)
    assert projected["algorithm"]["specification"]["steps"][0]["operation"] == "Buy a ship"
    monkeypatch.setattr("minecraft_mod_ai.typed_plan_authoring._extract_state_variable_types", lambda *_: {"credits": "int"})
    monkeypatch.setattr("minecraft_mod_ai.typed_plan_authoring.generate_fixed_template_value",
                        lambda *args, **kwargs: {"trigger_event": "command", "action_kind": "increment_state",
                                                 "state_key": "credits", "value": "1"})
    body = author_semantic_game_dispatch(
        object(), "Buy a ship", projected, {},
        coverage_refs=("algorithm.steps",), bound_events=("command",), record_action_guards=guards,
    )
    assert body[0]["condition"]["right"]["value"] == "ship_purchase"
    assert "upgrade" not in json.dumps(body)


def test_feature_execution_passes_materialized_owner_api_to_the_real_generator(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from minecraft_mod_ai.authored_feature_implementation import (
        compile_request_feature_jobs,
    )
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.host_version_catalog import host_target
    from minecraft_mod_ai.implementation_identity import compute_json_schema_hash
    from minecraft_mod_ai.integrity_dispatcher import execute_generator_job

    path = "src/main/java/test/space/PlanetRegistry.java"
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_text("package test.space; public class PlanetRegistry { "
                      "public static String destinationForShip(long id) { return \"planet\"; } }")
    content = SimpleNamespace(module_id="planet_registry", kind="item", config={"requirement_refs": ["planet"]})
    # This fixture tests dependency-file visibility, not template compilation:
    # represent an existing Java producer without inventing a canonical leaf.
    content_job = SimpleNamespace(
        owner_module="planet_registry", target_path=path, deterministic_inputs={},
    )
    adapter = host_target("26.1.2")
    plan = AuthoredPlan("Travel to planets", "Saved", request_scope=_scope_with_accepted_construction_behavior())
    modules, jobs = compile_request_feature_jobs(
        plan, adapter=adapter, mod_id="space", package_name="test.space",
        content_modules=(content,), content_jobs=(content_job,), runtime_modules=(),
        
    )
    assert "planet_registry" in modules[0].depends_on
    spec = jobs[0].deterministic_inputs["_canonical_inputs"]["event_callback_input"]
    contract_hash = compute_json_schema_hash(spec)

    class Router:
        def generate_text(self, role, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            assert "destinationForShip(long id)" in json.dumps(payload["materialized_prerequisite_source"])
            import hashlib

            registration = "registerBehavior" + hashlib.sha256(
                b"behavior_planet"
            ).hexdigest()[:20]
            return json.dumps(
                f"package {spec['bindings']['package_name']}; "
                f"public final class {spec['bindings']['class_name']} "
                "implements net.fabricmc.api.ModInitializer { "
                f"public void onInitialize() {{ this.{registration}(); }} "
                f"private void {registration}() {{ }} "
                "}"
            )

    monkeypatch.setattr("minecraft_mod_ai.integrity_validators.validate_java_syntax",
                        lambda *args, **kwargs: {"validator": "java_syntax", "status": "FIXTURE"})
    result = execute_generator_job(
        jobs[0], context={"resolved_version_context": adapter.version_context.to_dict()},
        router=Router(), base_dir=tmp_path,
    )
    assert compute_json_schema_hash(spec) == contract_hash
    assert result["canonical_output"]["event_callback_receipt"]["contract_sha256"] == contract_hash
    assert (tmp_path / spec["target_path"]).is_file()
    assert any(row["validator"] == "materialized_prerequisite_source" for row in result["validations"])


def test_feature_receives_real_template_registry_path_and_constant():
    from types import SimpleNamespace

    from minecraft_mod_ai.artifact_job import ArtifactJob
    from minecraft_mod_ai.authored_feature_implementation import (
        compile_request_feature_jobs,
    )
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.host_version_catalog import host_target

    owner = SimpleNamespace(module_id="ore", kind="item", config={"requirement_refs": ["planet"]})
    job = ArtifactJob("ore.registry", "fabric/item/register_keyed", "ore",
                      target_path="src/main/java/test/space/registry/ModItems.java",
                      deterministic_inputs={"java_constant": "ORE", "registry_path": "ore"})
    _, jobs = compile_request_feature_jobs(
        AuthoredPlan("Travel to planets", "Saved", request_scope=_scope_with_accepted_construction_behavior()),
        adapter=host_target("26.1.2"), mod_id="space", package_name="test.space",
        content_modules=(owner,), content_jobs=(job,), 
    )
    bindings = jobs[0].deterministic_inputs["_canonical_inputs"]["event_callback_input"]["bindings"]
    assert job.target_path in json.loads(bindings["content_dependency_paths_json"])
    assert {"package_name": "test.space.registry", "class_name": "ModItems",
            "java_constant": "ORE", "registry_path": "ore"} in json.loads(bindings["content_owner_classes_json"])


def test_four_runtime_owner_records_use_one_model_call(monkeypatch):
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    from minecraft_mod_ai.planner_budget import PlannerBudget

    design = sections()
    for index in range(3):
        design["algorithm"]["specification"]["steps"].append(
            {key: "Recruit crew member " + str(index)
             for key in DETAIL_RECORDS["algorithm"]["steps"].split()})
    ledger = scope()
    ledger["requirements"].append({
        "requirement_id": "crew", "statement": "Recruit crew",
        "implementation_role": "gameplay",
    })
    calls = []
    class Router:
        def generate_text(self, role, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            schema = kwargs["response_schema"]
            records = payload["records"]
            assert 1 <= len(records) <= 4
            assert len(schema["required"]) == len(records)
            calls.append(len(records))
            return json.dumps({
                row["key"]: ("planet" if row["record_ref"].startswith("integration.")
                             else "crew") for row in records
            })
    budget = PlannerBudget()
    owners = author_runtime_ownership(Router(), ledger, design, budget=budget)
    assert sum(calls) == 5
    assert calls == [4, 1]
    assert budget.call_count == 2
    grouped = {owner["feature_id"]: owner["worksheet_record_refs"] for owner in owners}
    assert len(grouped["planet"]) == 1
    assert len(grouped["crew"]) == 4


def test_invalid_runtime_owner_batch_never_silently_drops_record(monkeypatch):
    from minecraft_mod_ai.request_runtime_ownership import _choose_runtime_owner_batch

    class Router:
        def generate_text(self, role, messages, **kwargs):
            # The response is invalid even if the model can produce valid JSON.
            return json.dumps({"r0": "planet"})
    rows = [("algorithm.steps[" + str(i) + "]", {"operation": "attack"})
            for i in range(3)]
    features = [
        {"requirement_id": "planet", "statement": "Explore planets"},
        {"requirement_id": "crew", "statement": "Recruit crew"},
    ]
    from minecraft_mod_ai.structured_output import StructuredOutputValidationError

    with pytest.raises(StructuredOutputValidationError):
        _choose_runtime_owner_batch(Router(), rows, features, [], None)
