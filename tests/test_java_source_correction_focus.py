"""Production (19): a syntax reject must not replay an unchanged 50K prompt."""
from __future__ import annotations

import json
from types import SimpleNamespace

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
