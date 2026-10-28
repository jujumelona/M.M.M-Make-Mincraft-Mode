"""Unwritten requested UI effects reach source code through their original ABI."""
from __future__ import annotations

from copy import deepcopy

import pytest

from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
from test_typed_ui_action_authoring import sections


def source_sections():
    result = sections()
    result["resources_and_ui"]["specification"]["interactions"][0]["source_requirement_id"] = "selection"
    result["state_model"]["specification"]["updates"] = []
    return result


def source_capabilities():
    return {**typed_host_capability_contracts(), "ui.invoke_source_action": {
        "owner": "AuthoredSourceActions", "method": "invokeSourceAction",
        "parameters": ["string", "object", "map"], "return_type": "int", "gameplay_mutation": False}}


def test_source_ui_host_capability_accepts_its_map_abi_and_rejects_unknown_types():
    from minecraft_mod_ai.typed_plan_ir import validate_typed_host_capability_contracts
    caps = source_capabilities()
    assert validate_typed_host_capability_contracts(caps)["ui.invoke_source_action"]["parameters"][-1] == "map"
    caps["ui.invoke_source_action"]["parameters"][-1] = "untyped"
    with pytest.raises(ValueError, match="unsupported type"):
        validate_typed_host_capability_contracts(caps)


def test_missing_worksheet_mutation_creates_a_typed_source_call_without_model_repair():
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
    data = source_sections()
    before = deepcopy(data)
    functions, bindings = author_action_functions(
        None, "Implement the original selection", data, source_capabilities(), first_index=0, prefer_grounded=True)
    call = functions[0]["body"][0]["value"]
    assert call["op"] == "capability" and call["id"] == "ui.invoke_source_action"
    assert call["args"][1] == {"op": "ref", "name": "source"}
    assert call["args"][2]["entries"][0]["value"] == {"op": "ref", "name": "amount"}
    assert bindings[0]["config"]["action"]["inputs"] == data["resources_and_ui"]["specification"]["interactions"][0]["server_action"]["inputs"]
    validate_typed_plan_ir({"schema_version": "mmm/typed-plan-ir-v1", "source_sha256": "0" * 64,
        "functions": functions, "event_bindings": bindings, "initialize": []}, capabilities=source_capabilities())
    assert data == before


def test_source_ui_transport_is_not_labeled_as_an_implemented_gameplay_effect():
    from minecraft_mod_ai.typed_plan_ir import typed_plan_runtime_mutations, validate_typed_plan_ir
    from minecraft_mod_ai.typed_action_authoring import author_action_functions
    functions, bindings = author_action_functions(None, "Code the callback", source_sections(),
        source_capabilities(), first_index=0, prefer_grounded=True)
    plan = validate_typed_plan_ir({"schema_version": "mmm/typed-plan-ir-v1", "source_sha256": "0" * 64,
        "functions": functions, "event_bindings": bindings, "initialize": []}, capabilities=source_capabilities())
    assert typed_plan_runtime_mutations(plan) == ()


def request_scope(*, role="gameplay"):
    from minecraft_mod_ai.construction_task import BehaviorRequirement
    return {"requested_prompt": "Select an amount and record it.", "requirements": [{
        "requirement_id": "selection", "statement": "Select an amount and record it.",
        "implementation_role": role, "source_span": {"text": "Select an amount and record it."}}],
        "behavior_requirements": [BehaviorRequirement("selection.submit", "selection", "Record the requested amount",
            "Select an amount and record it.", "User submits the existing selection form", "Store the submitted amount").to_dict()]}


def test_production_action_preparation_keeps_missing_effect_as_code_work():
    from minecraft_mod_ai.complete_planner import prepare_production_actions
    data = source_sections()
    before = deepcopy(data)
    result = prepare_production_actions(None, data, request_scope(), minecraft_version="26.2")
    assert result == before
    assert result["state_model"]["specification"]["updates"] == []


@pytest.fixture(params=["missing_input", "incompatible_input_type"])
def invalid_source_guard_sections(request):
    from minecraft_mod_ai.authored_structured_design import normalize_structured_sections

    data = source_sections()
    action = data["resources_and_ui"]["specification"]["interactions"][0]
    input_name = "missing"
    if request.param == "incompatible_input_type":
        input_name = "amount"
        action["server_action"]["inputs"] = [{"name": "amount", "label": "Amount",
            "type": "string", "max_length": 10, "choices": []}]
    data["state_model"]["specification"]["transitions"] = [{
        "source_requirement_id": "selection", "trigger": action["trigger"],
        "from_state": "ready", "to_state": "ready", "mutation": [],
        "guard": {"kind": "compare", "op": ">",
            "left": {"kind": "context_ref", "name": input_name},
            "right": {"kind": "number", "value": "0"}},
    }]
    return normalize_structured_sections(data)


def test_production_preparation_rejects_invalid_deferred_guard_abi(invalid_source_guard_sections):
    from minecraft_mod_ai.complete_planner import prepare_production_actions

    with pytest.raises(ValueError, match="ACTION_UI_GUARD_ABI_UNBOUND"):
        prepare_production_actions(None, invalid_source_guard_sections, request_scope(),
                                   minecraft_version="26.2")


def test_direct_source_action_authoring_rejects_invalid_guard_abi(invalid_source_guard_sections):
    from minecraft_mod_ai.typed_action_authoring import author_action_functions

    with pytest.raises(ValueError, match="ACTION_UI_GUARD_ABI_UNBOUND"):
        author_action_functions(None, "Implement the original selection", invalid_source_guard_sections,
                                source_capabilities(), first_index=0, prefer_grounded=True)


def test_saved_bound_source_action_production_rejects_invalid_guard_abi(invalid_source_guard_sections):
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.authored_production import _compile_new_authored_modules
    from minecraft_mod_ai.production_state_compiler import compile_production_state_section
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    from minecraft_mod_ai.typed_action_authoring import migrate_ui_action_plan
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir

    # A saved delegate is already bound, so migration legitimately returns it unchanged.
    valid = deepcopy(invalid_source_guard_sections)
    valid["state_model"]["specification"]["transitions"] = []
    text = "Select an amount and record it."
    ir = author_typed_plan_ir(None, text, valid, source_capabilities(), deterministic_module_kinds=())
    owners = author_runtime_ownership(None, request_scope(), invalid_source_guard_sections)
    saved = AuthoredPlan(text, text, structured_sections=invalid_source_guard_sections,
        typed_plan_ir=ir, request_scope=request_scope(), content_design={"_request_runtime_owners": owners})
    assert migrate_ui_action_plan(None, saved, minecraft_version="26.2") is saved
    with pytest.raises(ValueError, match="ACTION_UI_GUARD_ABI_UNBOUND"):
        _compile_new_authored_modules(saved, mod_id="example", package_name="demo",
            target={"minecraft_version": "26.2"}, production_state_section=compile_production_state_section(saved))


@pytest.mark.parametrize("mutation", [[], [{"target": "amount", "operator": "=", "value": {"kind": "state_ref", "name": "amount"}}],
    [{"target": "amount", "operator": "+=", "value": {"kind": "number", "value": "0"}}]])
def test_source_noop_draft_is_retained_for_code_instead_of_becoming_a_typed_effect(mutation):
    from minecraft_mod_ai.complete_planner import prepare_production_actions
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    data = source_sections()
    trigger = data["resources_and_ui"]["specification"]["interactions"][0]["trigger"]
    data["state_model"]["specification"]["updates"] = [{"source_requirement_id": "selection",
        "trigger": trigger, "owner": "player", "mutation": deepcopy(mutation)}]
    prepared = prepare_production_actions(None, data, request_scope(), minecraft_version="26.2")
    owners = author_runtime_ownership(None, request_scope(), prepared)
    plan = author_typed_plan_ir(None, "Implement the requested effect", prepared, source_capabilities(),
        runtime_owners=owners, deterministic_module_kinds=())
    assert prepared == data
    assert plan["functions"][0]["body"][0]["value"]["id"] == "ui.invoke_source_action"


def test_real_typed_plan_keeps_original_action_without_inventing_a_mutation():
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
    from minecraft_mod_ai.typed_plan_support import typed_plan_support_issues
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    data = source_sections()
    owners = author_runtime_ownership(None, request_scope(), data)
    plan = author_typed_plan_ir(None, "Select an amount and record it.", data, source_capabilities(),
                              runtime_owners=owners, deterministic_module_kinds=())
    assert len(plan["functions"]) == len(plan["event_bindings"]) == 1
    assert plan["functions"][0]["body"][0]["value"]["id"] == "ui.invoke_source_action"
    assert typed_plan_support_issues(data, plan, runtime_owners=owners) == ()


def test_saved_production_plan_reaches_content_authoring_before_any_effect_repair(monkeypatch, tmp_path):
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
    from minecraft_mod_ai.host_version_catalog import host_target
    class ContentAuthoringReached(Exception):
        pass
    def content_boundary(*args, **kwargs):
        assert kwargs["request_catalog"]["requirements"]
        raise ContentAuthoringReached
    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr("minecraft_mod_ai.content_design_graph.compile_content_graph", content_boundary)
    data = source_sections()
    plan = AuthoredPlan("Select an amount and record it.", "Saved", structured_sections=data, request_scope=request_scope())
    planner = CompleteGameDesignPlanner(None, adapter=host_target("26.2"), deterministic_module_kinds=())
    with pytest.raises(ContentAuthoringReached):
        planner._author_for_production(plan)
    assert data["state_model"]["specification"]["updates"] == []


@pytest.mark.parametrize("role", ["gameplay", "interface"])
def test_empty_source_action_state_rule_belongs_to_feature_code(role):
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership
    data = source_sections()
    row = data["resources_and_ui"]["specification"]["interactions"][0]
    data["state_model"]["specification"]["updates"] = [{"owner": "player", "trigger": row["trigger"],
        "source_requirement_id": "selection", "mutation": []}]
    owners = author_runtime_ownership(None, request_scope(role=role), data)
    assert owners == [{"feature_id": "selection", "owner_module": "feature_selection",
                       "worksheet_record_refs": ["state_model.updates[0]"]}]


def test_interface_requirement_with_unwritten_action_still_schedules_java_body():
    import json
    from types import SimpleNamespace
    from minecraft_mod_ai.authored_feature_implementation import compile_request_feature_jobs
    from minecraft_mod_ai.authored_plan import AuthoredPlan
    from minecraft_mod_ai.host_version_catalog import host_target

    data = source_sections()
    plan = AuthoredPlan("Select an amount and record it.", "A source UI command", structured_sections=data,
                       request_scope=request_scope(role="interface"))
    gui = SimpleNamespace(module_id="panel", kind="gui", config={"requirement_refs": ["selection"]})
    modules, jobs = compile_request_feature_jobs(plan, adapter=host_target("26.2"),
        mod_id="example", package_name="demo", content_modules=[gui], content_jobs=[])
    assert len(modules) == len(jobs) == 1
    from minecraft_mod_ai.task_template_catalog import canonical_specification_input
    specification = canonical_specification_input(jobs[0].canonical_leaf, jobs[0].deterministic_inputs["_canonical_inputs"])
    callbacks = json.loads(specification["bindings"]["source_ui_callbacks_json"])
    assert len(callbacks) == 1 and callbacks[0]["source_requirement_id"] == "selection"
    assert callbacks[0]["return_type"] == "int"
    retained = json.loads(specification["bindings"]["retained_typed_behavior_json"])
    assert retained["source_ui_actions"][0]["implementation_status"] == "source_body_required"
    assert "src/main/java/demo/AuthoredSourceActions.java" in json.loads(specification["bindings"]["required_content_source_paths_json"])
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    builder = IncrementalJavaBuilder(None, specification, [], {"symbols": {}},
                                    json.loads(specification["bindings"]["construction_behaviors_json"]))
    assert builder.pending[0]["event"]["action_id"] == callbacks[0]["action_id"]


def test_source_ui_transport_is_materialized_before_feature_code_authoring(tmp_path):
    import json
    from types import SimpleNamespace
    from minecraft_mod_ai.authored_production import materialize_authored_execution_scaffold, _sha256_json
    from minecraft_mod_ai.typed_plan_authoring import author_typed_plan_ir
    data = source_sections()
    plan = author_typed_plan_ir(None, "Select an amount and record it.", data, source_capabilities(), deterministic_module_kinds=())
    source_path = "src/main/java/demo/AuthoredSourceActions.java"
    manifest = {"schema_version": "mmm/authored-execution-manifest-v2", "policy": "host_typed_plan_ir",
        "source_text_sha256": plan["source_sha256"], "typed_program": {
            "path": "src/main/java/demo/AuthoredProgram.java", "capabilities": ["ui.invoke_source_action"],
            "state_required": False, "dry_compile": {"paths": [source_path]}}}
    manifest["manifest_sha256"] = _sha256_json(manifest)
    proposal = SimpleNamespace(game_design={"_authored_execution_manifest": manifest,
        "authored_plan": {"typed_plan_ir": plan}}, base_proposal=SimpleNamespace(spec=SimpleNamespace(
            package_name="demo", platform=SimpleNamespace(minecraft_version="26.2"))))
    metadata = tmp_path / "src/main/resources/fabric.mod.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({"schemaVersion": 1, "id": "example", "version": "1", "entrypoints": {"main": ["demo.Mod"]}}))
    main = tmp_path / "src/main/java/demo/Mod.java"
    main.parent.mkdir(parents=True)
    main.write_text("package demo; public class Mod implements net.fabricmc.api.ModInitializer {public void onInitialize(){}}")
    materialize_authored_execution_scaffold(proposal, tmp_path)
    target = tmp_path / source_path
    assert target.is_file()
    assert "EventFactory.createArrayBacked" in target.read_text()
    before = target.read_text()
    materialize_authored_execution_scaffold(proposal, tmp_path)
    assert target.read_text() == before
    target.write_text("package demo; public class AuthoredSourceActions {}")
    with pytest.raises(ValueError, match="OWNERSHIP"):
        materialize_authored_execution_scaffold(proposal, tmp_path)
