"""Regression for Colab 26.2 failures: 46K-token Java prompts and duplicate repair ABI.

The authoritative compiler retains the whole project/state model. Only the
current behavior's exact Java accessor signatures belong in the model prompt.
"""
import json
from dataclasses import dataclass

from minecraft_mod_ai.construction_java_context import (
    scoped_java_member_messages,
    ensure_live_java_source_capacity,
)


def _fixture():
    owner = "example.AuthoredStateModel"
    methods, headers = [], []
    for idx in range(75):
        name = "state_" + str(idx)
        for prefix in ("readState_", "writeState_"):
            method = prefix + name
            signature = ("public static String " if prefix.startswith("read")
                         else "public static void ") + method + "();"
            methods.append({"name": method, "signature": signature})
            headers.append(signature)
    frame = {
        "kind": "method", "prefix": "private void handleBehavior() {\n",
        "suffix": "\n}", "host_state_access": [
            {"owner": owner, "methods": methods},
        ],
        "host_behavior_contract": {
            "behavior_id": "beh-1", "state_reads": ["state_3"],
            "state_writes": ["state_3"],
        },
    }
    prerequisite = [
        {"path": "src/State.java", "types": [{"owner": owner,
            "declaration": "public class AuthoredStateModel", "members": headers}]},
        {"path": "src/Other.java", "types": [{"owner": "example.Other",
            "declaration": "public class Other", "members": ["public void realMethod();"]}]},
    ]
    current = {
        "requirement": "Implement the original interaction",
        "response_source_frame": frame, "assigned_behavior": {
            "behavior_id": "beh-1", "intent": "Original interaction",
            "effects": "Apply original effects", "state_reads": ["state_3"],
            "state_writes": ["state_3"],
        },
        "host_state_work_order_scope": {
            "declared_readable": ["state_" + str(i) for i in range(75)],
            "declared_writable": ["state_" + str(i) for i in range(75)],
            "undeclared_plan_reads": [], "undeclared_plan_writes": [],
        },
    }
    messages = [
        {"role": "system", "content": "Write only the current Java member"},
        {"role": "user", "content": json.dumps({
            "assigned_behaviors": [current["assigned_behavior"]],
            "inspected_target_api": {"class_members": []},
            "materialized_prerequisite_api": prerequisite,
        })},
        {"role": "user", "content": json.dumps(current)},
    ]
    return messages, frame, prerequisite


def test_model_sees_only_its_owned_state_accessors_not_75_unrelated_variables():
    original, frame, prerequisites = _fixture()
    scoped = scoped_java_member_messages(original, frame)
    task = json.loads(scoped[-1]["content"])
    assert len(task["host_state_access"]) == 1
    methods = task["host_state_access"][0]["methods"]
    assert {method["name"] for method in methods} == {
        "readState_state_3", "writeState_state_3",
    }
    assert all("state_4" not in m["signature"] for m in methods)
    assert task["host_state_work_order_scope"]["declared_readable"] == ["state_3"]
    assert task["host_state_work_order_scope"]["declared_writable"] == ["state_3"]
    assert task["materialized_prerequisite_api"][0]["types"][0]["members"] == [
        "public static String readState_state_3();",
        "public static void writeState_state_3();",
    ]
    assert task["materialized_prerequisite_api"][1] == prerequisites[1]
    assert len(scoped[-1]["content"]) < len(original[-1]["content"]) + 4000
    assert len(frame["host_state_access"][0]["methods"]) == 150


def test_parser_correction_does_not_echo_entire_state_contract_twice():
    original, frame, prerequisites = _fixture()
    previous = 'private void otherMethod() {}'
    correction = {
        "immutable_source_frame": frame,
        "assigned_behavior": original and json.loads(original[-1]["content"])["assigned_behavior"],
        "api_evidence": {
            "selected_target_api": [], "typed_input_producers": [],
            "class_members": [], "materialized_prerequisite_api": prerequisites,
        },
        "previous_implementation": previous,
        "diagnostic": "missing syntax at 3:48",
        "host_state_work_order_scope": json.loads(original[-1]["content"])["host_state_work_order_scope"],
    }
    scoped = scoped_java_member_messages(
        [{"role": "user", "content": json.dumps(correction)}], frame)
    task = json.loads(scoped[-1]["content"])
    assert "immutable_source_frame" not in task
    assert "response_source_frame" not in task
    assert task["previous_implementation"] == previous
    assert task["diagnostic"] == "missing syntax at 3:48"
    assert len(task["host_state_access"][0]["methods"]) == 2
    assert len(task["api_evidence"]["materialized_prerequisite_api"][0]["types"][0]["members"]) == 2
    assert len(scoped[-1]["content"]) < len(json.dumps(correction)) // 3


def test_exact_context_preflight_projects_member_before_budget_is_exhausted(monkeypatch):
    original, frame, _ = _fixture()

    @dataclass(frozen=True)
    class Request:
        messages: tuple

    class Config:
        adapter = "llama_cpp"

    class Adapter:
        def input_context_accounting(self, request):
            size = len(json.dumps(request.messages))
            return type("Count", (), {
                "input_tokens": size // 3,
                "context_tokens": 32768,
                "remaining_tokens": 32768 - size // 3,
            })()

    monkeypatch.setattr(
        "minecraft_mod_ai.model_context_budget.fit_messages_to_context",
        lambda messages, config, tools: tuple(messages),
    )
    request = Request(tuple(original))
    result = ensure_live_java_source_capacity(Adapter(), Config(), request, original, frame)
    assert result.messages != request.messages
    task = json.loads(result.messages[-1]["content"])
    assert len(task["host_state_access"][0]["methods"]) == 2
