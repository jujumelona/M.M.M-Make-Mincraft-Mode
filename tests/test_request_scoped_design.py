"""A bounded worksheet must not turn a multi-action request into one action."""
from copy import deepcopy
import json
import re

import pytest

from minecraft_mod_ai.authored_structured_design import normalize_structured_sections
from minecraft_mod_ai.complete_planner import CompleteGameDesignPlanner
from minecraft_mod_ai.planning_detail_slots import DETAIL_RECORDS
from worksheet_fixtures import _fixture_value


ACTIONS = ("mine", "trade", "craft", "launch", "fight", "colonize")
PROMPT = "Add actions: " + ", ".join(ACTIONS)


def scope():
    return {
        "schema_version": "mmm/request-scope-v1",
        "requested_prompt": PROMPT,
        "requirements": [
            {"requirement_id": action, "statement": action,
             "source_span": {"text": action}, "implementation_role": "gameplay",
             "search_query": action, "rag_references": []}
            for action in ACTIONS
        ],
    }


def empty_section(section):
    return {"constraint_evidence_refs": [], "specification": {
        **{key: [] for key in DETAIL_RECORDS[section]}, "inapplicable_concerns": [],
    }}


class FeatureRouter:
    """External inference fixture: global requests collapse to launch, as in log 31."""
    def __init__(self, *, fail_action=""):
        self.fail_action = fail_action
        self.started = []

    def generate_text(self, role, messages, **kwargs):
        content = "\n".join(row["content"] for row in messages)
        match = re.search(r'ACTIVE REQUEST FEATURE:\n([^\n]+)', content)
        action = json.loads(match[1])["requirement_id"] if match else "launch"
        schema = kwargs["response_schema"]
        fields = schema["properties"]
        if "Concern fields:" in content:
            if "Section: behavior_contract\n" in content:
                if action == self.fail_action:
                    raise RuntimeError("simulated inference interruption")
                self.started.append(action)
            return json.dumps({name: value["enum"][0] for name, value in fields.items()})
        result = _fixture_value(schema, action)
        if "entry_points" in result:
            for row in result["entry_points"]:
                if "trigger" in row:
                    row["trigger"] = "command:" + action
        return json.dumps(result)


def test_real_planner_keeps_every_source_feature_in_observable_design(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr("minecraft_mod_ai.request_scope.author_request_scope",
                        lambda *a, **k: scope())
    planned = CompleteGameDesignPlanner(FeatureRouter()).plan(PROMPT)
    behavior = planned.structured_sections["behavior_contract"]["specification"]
    assert len(behavior["entry_conditions"]) == 6
    outputs = " ".join(row["name"] for row in behavior["outputs"])
    assert all(action in outputs for action in ACTIONS)
    assert planned.request_scope == scope()


def test_canonical_accumulation_keeps_more_than_one_model_page_of_updates():
    section = empty_section("state_model")
    section["specification"]["variables"] = [{
        "name": "credits", "type": "int", "default": "0", "owner": "player",
        "unit": "credits", "domain": "nonnegative",
    }]
    section["specification"]["updates"] = [
        {"trigger": action, "owner": "player", "mutation": [{
            "target": "credits", "operator": "+=",
            "value": {"kind": "number", "value": "1"},
        }]} for action in ACTIONS
    ]
    result = normalize_structured_sections({"state_model": section})
    assert [row["trigger"] for row in result["state_model"]["specification"]["updates"]] == list(ACTIONS)


def test_conflicting_shared_state_definition_is_never_silently_overwritten():
    from minecraft_mod_ai.request_structured_design import merge_request_sections

    prior = {"state_model": empty_section("state_model")}
    prior["state_model"]["specification"]["variables"] = [{
        "name": "credits", "type": "int", "default": "0", "owner": "player",
        "unit": "credits", "domain": "nonnegative",
    }]
    later = deepcopy(prior)
    later["state_model"]["specification"]["variables"][0]["type"] = "long"
    before = deepcopy(prior)
    with pytest.raises(ValueError, match="REQUEST_STATE_DEFINITION_CONFLICT.*credits"):
        merge_request_sections(prior, later)
    assert prior == before


def test_interrupted_request_resumes_without_regenerating_accepted_features(monkeypatch, tmp_path):
    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr("minecraft_mod_ai.request_scope.author_request_scope",
                        lambda *a, **k: scope())
    router = FeatureRouter(fail_action="fight")
    planner = CompleteGameDesignPlanner(router)
    with pytest.raises(RuntimeError, match="simulated inference interruption"):
        planner.plan(PROMPT)
    router.fail_action = ""
    planned = planner.plan(PROMPT)
    assert len(planned.structured_sections["behavior_contract"]["specification"]["outputs"]) == 6
    assert router.started == list(ACTIONS)


def test_feature_checkpoint_is_bound_to_shared_state_inputs(tmp_path, monkeypatch):
    from minecraft_mod_ai.planner_checkpoint import PlannerCheckpoint

    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path))
    feature = scope()["requirements"][0]
    checkpoint = PlannerCheckpoint(prompt=PROMPT)
    accepted = {"state_model": empty_section("state_model")}
    checkpoint.save_request_design(feature, [], accepted)
    restored = PlannerCheckpoint(prompt=PROMPT)
    assert restored.load_request_design(feature, []) == accepted
    assert restored.load_request_design(feature, [{"name": "new_state"}]) is None


class SharedStateRouter(FeatureRouter):
    """Six transactions share one declaration, authored only by the first feature."""
    def generate_text(self, role, messages, **kwargs):
        content = "\n".join(row["content"] for row in messages)
        match = re.search(r'ACTIVE REQUEST FEATURE:\n([^\n]+)', content)
        action = json.loads(match[1])["requirement_id"] if match else "launch"
        fields = kwargs["response_schema"]["properties"]
        if "Concern fields:" in content:
            result = json.loads(super().generate_text(role, messages, **kwargs))
            if "Section: state_model\n" in content:
                result.update(variables="1" if action == "mine" else "0", updates="1")
            if "Section: resources_and_ui\n" in content:
                result["interactions"] = "1"
            return json.dumps(result)
        if "Target: state_model.variables[" in content:
            variable = {"name": "credits", "type": "int", "default": "0",
                        "owner": "player", "unit": "credits", "domain": "nonnegative"}
            return json.dumps({field: variable[field] for field in fields})
        if "Target: state_model.updates[" in content:
            update = {"owner": "player", "trigger": action}
            return json.dumps({field: update[field] for field in fields})
        if "count" in fields:
            return json.dumps({"count": "one"})
        for field, value in (("target", "credits"), ("operator", "+="),
                             ("source", "number"), ("value", "1")):
            if field in fields:
                return json.dumps({field: value})
        if "interactions" in fields:
            projected = fields["interactions"]["items"]["properties"]
            record = {"trigger": action, "client_feedback": "Show updated credits",
                      "server_action": "Collect credits: " + action}
            return json.dumps({"interactions": [{name: record[name] for name in projected}]})
        return super().generate_text(role, messages, **kwargs)


def test_shared_state_transactions_survive_design_and_source_owned_lowering(monkeypatch, tmp_path):
    from minecraft_mod_ai.typed_action_authoring import (
        author_action_functions, preflight_ui_action_effects, promote_action_sections,
    )
    from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
    from minecraft_mod_ai.request_runtime_ownership import author_runtime_ownership

    monkeypatch.setenv("MMM_PLAN_CHECKPOINT_DIR", str(tmp_path))
    monkeypatch.setattr("minecraft_mod_ai.request_scope.author_request_scope",
                        lambda *a, **k: scope())
    planned = CompleteGameDesignPlanner(SharedStateRouter()).plan(PROMPT)
    state = planned.structured_sections["state_model"]["specification"]
    assert [row["name"] for row in state["variables"]] == ["credits"]
    assert [row["trigger"] for row in state["updates"]] == list(ACTIONS)
    promoted = promote_action_sections(None, planned.structured_sections)
    preflight_ui_action_effects(promoted, strict_source=True)
    owners = author_runtime_ownership(None, scope(), promoted)
    functions, bindings = author_action_functions(
        None, PROMPT, promoted, typed_host_capability_contracts(),
        first_index=6, prefer_grounded=True, runtime_owners=owners,
    )
    assert len(functions) == len(bindings) == 6
    for function in functions:
        writes = [node for node in function["body"] if node["op"] == "state_set"]
        assert len(writes) == 1
        assert writes[0]["key"] == {"op": "literal", "type": "string", "value": "credits"}
