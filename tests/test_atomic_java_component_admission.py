"""Production native response -> component admission -> real Java execution."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator

from minecraft_mod_ai.atomic_concern_source import AtomicConcernExecutor
from minecraft_mod_ai.atomic_java_admission import rejected_decision
from minecraft_mod_ai.custom_module_errors import (
    AtomicJavaDecisionError,
    CustomModuleGenerationError,
)
from minecraft_mod_ai.custom_module_generator import (
    _atomic_parameters_for_request,
    _call_atomic_java_region,
)
from minecraft_mod_ai.model_adapters.base import NativeToolDecisionRejected


def messages():
    return [{"role": "user", "content": json.dumps({
        "response_region": "members", "host_selected_class": "AuthoredStateModel",
        "concern": {"name": "variables"},
    })}]


class Router:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_tool_decision(self, role, request, **kwargs):
        self.calls.append((deepcopy(request), deepcopy(kwargs)))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return deepcopy(value)


def executor(tmp_path, call):
    return AtomicConcernExecutor(
        root=tmp_path, target=tmp_path / "AuthoredStateModel.java",
        relative="AuthoredStateModel.java", symbol="AuthoredStateModel",
        original="// MMM_AUTHORED_FEATURE_BODY\n", task={"task_id": "test"},
        section="state_model", concerns=({"sequence": 0, "identifier": "variables",
            "concern": "variables", "task": "state", "rules": []},),
        grounding={}, dependency_source="", require_initialize=False, call_coder=call,
        compile_java=lambda _: SimpleNamespace(status="PASS"), compile_log=lambda _: "",
        write_source=lambda *_: None,
    )


def test_logged_outer_class_collision_repairs_only_rejected_component_and_runs_java(tmp_path):
    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("Java compiler/runtime unavailable")
    bad = {
        "fields": [{"name": "credits", "type": "int", "initializer": "7"}],
        "classes": [{"name": "AuthoredStateModel", "fields": [{"name": "fuel", "type": "int"}]}],
        "methods": [{"modifiers": ["public"], "name": "spend", "return_type": "boolean",
                     "parameters": [{"type": "int", "name": "cost"}],
                     "body": ["if (cost < 0 || cost > credits) return false", "credits -= cost", "return true"]}],
    }
    before = deepcopy(bad)
    router = Router([bad, {"value": "ShipData"}])
    run = executor(tmp_path, lambda request: _call_atomic_java_region(router, request, output_token_ceiling=None))
    source = run.run()["source"]
    assert bad == before
    assert [kwargs["tool_name"] for _, kwargs in router.calls] == ["emit_java_structure", "repair_java_component"]
    correction = json.loads(router.calls[1][0][-1]["content"])["component_repair"]
    assert correction["category"] == "classes"
    assert correction["selected_path"] == ["name"]
    assert router.calls[1][1]["parameters"]["properties"]["value"]["type"] == "string"
    assert correction["accepted_components"][0]["declaration"]["name"] == "credits"
    assert "class ShipData" in source
    assert "int fuel;" in source
    repair_payload = json.loads(router.calls[1][0][-1]["content"])
    assert repair_payload["scope"]["required_output_tool"] == "repair_java_component"
    assert "generation_recipe" not in repair_payload
    assert source.count("class AuthoredStateModel") == 1
    target = tmp_path / "AuthoredStateModel.java"
    target.write_text(source, encoding="utf-8")
    probe = tmp_path / "Probe.java"
    probe.write_text('''public class Probe {
        public static void main(String[] args) {
            if (!AuthoredStateModel.spend(5)) throw new AssertionError("purchase");
            if (AuthoredStateModel.spend(3)) throw new AssertionError("overspend");
            if (AuthoredStateModel.spend(-1)) throw new AssertionError("negative");
            if (!AuthoredStateModel.spend(2)) throw new AssertionError("balance changed on rejection");
            System.out.println("PASS");
        }
    }''', encoding="utf-8")
    compile_result = subprocess.run([javac, str(target), str(probe)], capture_output=True, text=True, check=False)
    assert compile_result.returncode == 0, compile_result.stderr
    result = subprocess.run([java, "-cp", str(tmp_path), "Probe"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PASS"


def test_native_schema_prevents_host_name_and_salvages_its_rejected_payload():
    payload = json.loads(messages()[0]["content"])
    parameters, _ = _atomic_parameters_for_request(payload, response_region="members")
    bad = {"classes": [{"name": "AuthoredStateModel"}], "fields": [{"name": "credits", "type": "int"}]}
    assert not Draft202012Validator(parameters).is_valid(bad)
    rejection = NativeToolDecisionRejected("emit_java_structure", [{
        "original_tool": "emit_java_structure", "failure_code": "TOOL_SCHEMA_INVALID",
        "raw_arguments": json.dumps(bad), "error": "host class name is reserved",
    }])
    router = Router([rejection, {"value": "ShipData"}])
    source = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert "class ShipData" in source and "static int credits;" in source


@pytest.mark.parametrize("category,item", [
    ("fields", {"name": "credits", "type": 5}),
    ("methods", {"name": "value", "return_type": False}),
])
def test_schema_error_repairs_one_member_without_replacing_good_sibling(category, item):
    good = {"name": "other", "type": "int", "initializer": "42"}
    bad = {"fields": [good]}
    bad.setdefault(category, []).append(item)
    router = Router([bad, {"value": "int"}])
    result = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert "static int other = 42;" in result
    schema = router.calls[1][1]["parameters"]
    assert set(schema["properties"]) == {"value"}
    assert schema["additionalProperties"] is False


def test_repeated_invalid_component_is_terminal_without_regenerating_region(tmp_path):
    bad = {"name": "AuthoredStateModel"}
    router = Router([{"classes": [bad]}, {"value": "AuthoredStateModel"}])
    run = executor(tmp_path, lambda request: _call_atomic_java_region(router, request, output_token_ceiling=None))
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_COMPONENT_REPAIR_NO_PROGRESS") as error:
        run.run()
    assert len(router.calls) == 2
    assert error.value.response_sha256 != hashlib.sha256(b"").hexdigest()
    assert "AuthoredStateModel" in error.value.response_text


def test_changed_rejected_responses_do_not_become_identical_empty_source(tmp_path):
    responses = iter([
        AtomicJavaDecisionError("ATOMIC_CONCERN_RESPONSE_INVALID: malformed", response={"fields": "first"}),
        AtomicJavaDecisionError("ATOMIC_CONCERN_RESPONSE_INVALID: malformed", response={"fields": "second"}),
        "private static int value = 1;",
    ])
    def call(_):
        answer = next(responses)
        if isinstance(answer, Exception):
            raise answer
        return answer
    assert "value = 1" in executor(tmp_path, call).run()["source"]


def test_absent_response_is_not_proof_of_repeated_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "3")
    calls = []
    def call(_):
        calls.append(1)
        raise CustomModuleGenerationError("ATOMIC_CONCERN_RESPONSE_INVALID: no response")
    with pytest.raises(CustomModuleGenerationError, match="RETRY_EXHAUSTED"):
        executor(tmp_path, call).run()
    assert len(calls) == 3


@pytest.mark.parametrize("field,value", [
    ("original_tool", "unrelated_tool"), ("failure_code", "TOOL_NOT_VISIBLE"),
    ("raw_arguments", "not json"), ("raw_arguments", "[]"),
])
def test_does_not_salvage_unrelated_native_rejection(field, value):
    rejection = {"original_tool": "emit_java_structure", "failure_code": "TOOL_SCHEMA_INVALID",
                 "raw_arguments": '{"classes": []}'}
    rejection[field] = value
    assert rejected_decision(NativeToolDecisionRejected("emit_java_structure", [rejection]), "emit_java_structure") is None


def test_component_repair_budget_stops_distinct_invalid_answers(monkeypatch):
    monkeypatch.setenv("MMM_ATOMIC_CONCERN_REGION_ATTEMPTS", "2")
    router = Router([{"fields": [{"name": "x", "type": 1}]},
                     {"value": 2},
                     {"value": 3}])
    with pytest.raises(AtomicJavaDecisionError, match="ATOMIC_COMPONENT_REPAIR_NO_PROGRESS"):
        _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert len(router.calls) == 3


def test_cross_category_duplicate_type_is_repaired_without_touching_first_type():
    router = Router([{"records": [{"name": "Data"}], "classes": [{"name": "Data"}]},
                     {"value": "ShipData"}])
    result = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert "record Data" in result and "class ShipData" in result


def test_null_response_and_missing_response_have_different_evidence():
    missing = AtomicJavaDecisionError("missing")
    null = AtomicJavaDecisionError("invalid null", response=None)
    assert missing.response_sha256 is None
    assert null.response_text == "null"
    assert null.response_sha256 == hashlib.sha256(b"null").hexdigest()


def test_forward_sibling_declarations_are_visible_during_repair():
    router = Router([{"classes": [{"name": "AuthoredStateModel"}],
                      "fields": [{"type": "int", "name": "credits", "initializer": "7"}]},
                     {"value": "ShipData"}])
    _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    correction = json.loads(router.calls[1][0][-1]["content"])["component_repair"]
    assert correction["other_component_declarations"][0]["declaration"]["name"] == "credits"
    assert "initializer" not in correction["other_component_declarations"][0]["declaration"]


def test_recursive_type_name_repair_includes_its_self_references():
    router = Router([
        {"classes": [{"name": "AuthoredStateModel", "fields": [
            {"type": "AuthoredStateModel", "name": "next"}]}],
         "fields": [{"type": "int", "name": "credits", "initializer": "7"}]},
        {"value": {"name": "ShipData", "fields": [{"type": "ShipData", "name": "next"}]}},
    ])
    source = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    correction = json.loads(router.calls[1][0][-1]["content"])["component_repair"]
    assert correction["selected_path"] == []
    assert router.calls[1][1]["parameters"]["properties"]["value"]["type"] == "object"
    assert "ShipData next;" in source
    assert "AuthoredStateModel next;" not in source
    assert "int credits = 7;" in source


def test_class_name_in_string_literal_does_not_expand_name_repair_scope():
    router = Router([
        {"classes": [{"name": "AuthoredStateModel", "fields": [
            {"type": "String", "name": "label", "initializer": '"AuthoredStateModel"'}]}]},
        {"value": "ShipData"},
    ])
    source = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    correction = json.loads(router.calls[1][0][-1]["content"])["component_repair"]
    assert correction["selected_path"] == ["name"]
    assert '"AuthoredStateModel"' in source


@pytest.mark.parametrize("name", [[], {}, None, 12])
def test_invalid_name_type_repairs_value_without_losing_class_body(name):
    router = Router([
        {"classes": [{"name": name, "fields": [{"type": "int", "name": "fuel"}]}]},
        {"value": "ShipData"},
    ])
    source = _call_atomic_java_region(router, messages(), output_token_ceiling=None)
    assert "class ShipData" in source
    assert "int fuel;" in source
    assert json.loads(router.calls[1][0][-1]["content"])["component_repair"]["selected_path"] == ["name"]
