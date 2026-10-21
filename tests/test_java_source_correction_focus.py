"""Production (19): a syntax reject must not replay an unchanged 50K prompt."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.construction_source_repair import focused_java_correction_messages
from minecraft_mod_ai.model_adapters.base import AdapterConfig

FRAME = {
    "kind": "method", "prefix": "private boolean check(Object value) {\n",
    "suffix": "\n}",
}


def test_focused_retry_preserves_current_contract_not_unrelated_owner_catalog():
    huge = "UNRELATED_OWNER_MEMBER_" * 2600
    source_context = {"role": "user", "content": json.dumps({
        "assigned_behaviors": [{"behavior_id": "a", "intent": "Real gameplay"}],
        "inspected_target_api": {
            "target_api": [{"owner": "example.RealApi", "name": "run"}],
            "class_members": [{"members": [{"description": huge}]}],
        },
        "materialized_prerequisite_api": [{"types": [{"owner": "example.Model",
            "members": ["public static boolean readState_ready();"]}]}],
    })}
    task = {"role": "user", "content": json.dumps({
        "response_source_frame": FRAME,
        "current_unit_purpose": "Verify gameplay event",
        "assigned_behavior": {"behavior_id": "a", "intent": "Real gameplay"},
        "declared_member_api": [{"signature": "private boolean check(Object value)"}],
        "host_signature": "private boolean check(Object value)",
    })}
    messages = [source_context, task]
    focused = focused_java_correction_messages(
        messages, frame=FRAME,
        response=json.dumps({"implementation": "return value instanceof Sample.create();"}),
        diagnostic="missing syntax at 3:48",
    )
    payload = json.loads(focused[-1]["content"])
    assert len(focused[-1]["content"]) < 6000
    assert huge not in focused[-1]["content"]
    assert payload["assigned_behavior"]["intent"] == "Real gameplay"
    assert payload["api_evidence"]["selected_target_api"][0]["name"] == "run"
    assert payload["api_evidence"]["materialized_prerequisite_api"]
    assert payload["previous_implementation"] == "return value instanceof Sample.create();"
    assert payload["immutable_source_frame"] == FRAME


def test_java_source_router_repairs_parse_failure_with_focused_second_turn(monkeypatch):
    from test_first_pass_java_data_transport import DataRouter

    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda messages, **kwargs: messages,
    )
    calls = []

    def generate(request):
        calls.append(request)
        body = ('return value instanceof java.lang.String.valueOf("x");'
                if len(calls) == 1 else "return value instanceof java.lang.String;")
        return json.dumps({"implementation": body})

    adapter = SimpleNamespace(
        config=AdapterConfig(role="coder", adapter="llama_cpp"), generate=generate,
    )
    long_info = "UNRELATED_MODEL_INVENTORY_" * 2200
    msgs = [
        {"role": "user", "content": json.dumps({
            "inspected_target_api": {"target_api": [], "class_members": [long_info]},
            "assigned_behaviors": [{"behavior_id": "test", "intent": "predicate"}],
        })},
        {"role": "user", "content": json.dumps({
            "response_source_frame": FRAME,
            "current_unit_purpose": "Match a type using a valid Java expression",
            "assigned_behavior": {"behavior_id": "test", "intent": "predicate"},
            "host_signature": "private boolean check(Object value)",
        })},
    ]
    result = DataRouter(adapter).generate_java_source(
        "coder", msgs, data_name="test_parse_recovery", source_frame=FRAME,
    )
    assert "value instanceof java.lang.String;" in result
    assert len(calls) == 2
    second = calls[1].messages
    assert "UNRELATED_MODEL_INVENTORY" not in str(second)
    assert len(str(second)) < len(str(calls[0].messages)) / 3


def test_surplus_body_brace_does_not_hide_the_actual_state_abi_failure():
    from minecraft_mod_ai.construction_source_repair import assemble_java_implementation

    frame = {**FRAME, "host_state_access": [{
        "owner": "example.AuthoredStateModel",
        "methods": [{"name": "readState_balance", "signature": "public static int readState_balance();"}],
    }]}
    response = json.dumps({"implementation":
        'return AuthoredStateModel.getState("missing") != null;\n}'})
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS") as caught:
        assemble_java_implementation(response, frame)
    assert caught.value.failed_response == response
    assert caught.value.failed_source == (
        FRAME["prefix"] + 'return AuthoredStateModel.getState("missing") != null;' + FRAME["suffix"]
    )


def test_focused_retry_keeps_verified_receiver_and_callback_result_contracts():
    receiver = {"verified_class": "example.Player", "members": [{
        "owner": "example.Player", "name": "getUUID", "parameters": [],
        "return_type": "java.util.UUID", "kind": "METHOD",
    }]}
    callback = {"verified_class": "example.Result", "members": [{
        "owner": "example.Result", "name": "PASS", "kind": "FIELD",
        "return_type": "example.Result", "static": True,
    }]}
    messages = [{"role": "user", "content": json.dumps({
        "inspected_target_api": {"target_api": [{
            "owner": "example.Player", "name": "Player", "kind": "CLASS",
        }], "class_members": [receiver]},
    })}, {"role": "user", "content": json.dumps({
        "response_source_frame": FRAME, "callback_input_api": [callback],
    })}]
    focused = focused_java_correction_messages(messages, frame=FRAME,
        response='{"implementation":"return bad;"}', diagnostic="cannot find symbol")
    payload = json.loads(focused[-1]["content"])
    assert payload["callback_input_api"] == [callback]
    assert payload["api_evidence"]["class_members"][0]["members"] == receiver["members"]


def test_focused_correction_receives_state_error_after_host_brace_normalization(monkeypatch):
    from test_first_pass_java_data_transport import DataRouter

    monkeypatch.setattr("minecraft_mod_ai.model_context_budget.fit_messages_to_context",
                        lambda messages, **kwargs: messages)
    calls = []
    frame = {**FRAME, "host_state_access": [{
        "owner": "example.AuthoredStateModel", "methods": [{"name": "readState_balance"}],
    }]}

    def generate(request):
        calls.append(request)
        if len(calls) == 1:
            body = 'return AuthoredStateModel.getState("balance") != null;\n}'
        else:
            payload = json.loads(request.messages[-1]["content"])
            assert "JAVA_HOST_RAW_STATE_ACCESS" in payload["diagnostic"]
            assert "readState_balance" in str(payload["immutable_source_frame"])
            body = "return AuthoredStateModel.readState_balance() > 0;"
        return json.dumps({"implementation": body})

    adapter = SimpleNamespace(config=AdapterConfig(role="coder", adapter="llama_cpp"),
                              generate=generate)
    source = DataRouter(adapter).generate_java_source(
        "coder", [], data_name="state_failure", source_frame=frame)
    assert "AuthoredStateModel.readState_balance() > 0;" in source
    assert len(calls) == 2
