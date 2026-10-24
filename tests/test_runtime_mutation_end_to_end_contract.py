"""Native Minecraft mutation work cannot disappear between planning and Java."""

import json

import pytest

from minecraft_mod_ai.feature_execution_route import select_feature_execution_route
from minecraft_mod_ai.feature_source_work_graph import build_source_work_graph, member_source_work
from minecraft_mod_ai.host_callback_body import author_callback_body
from minecraft_mod_ai.runtime_mutation_author import mutation_targets


class Router:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((data_name, json.loads(messages[-1]["content"]), parameters))
        return next(self.responses)


def verified_frame():
    return {
        "kind": "method",
        "prefix": "private boolean apply(example.Target target, example.Hand hand) {\n",
        "suffix": "\n}",
        "host_behavior_contract": {
            "behavior_id": "gameplay-native",
            "state_reads": [], "state_writes": [],
            "runtime_state_writes": ["Player health"],
            "intent": "Apply a verified world-owned operation",
        },
    }


def verified_messages():
    return [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "parameters": ["example.Target", "example.Hand"],
        },
        "callback_input_api": [{
            "verified_class": "example.Target", "members": [
                {"kind": "METHOD", "name": "apply", "static": False,
                 "return_type": "void", "parameters": ["example.Hand"]},
                {"kind": "METHOD", "name": "getState", "static": False,
                 "return_type": "int", "parameters": []},
            ],
        }],
    })}]


def test_native_write_forces_source_owned_even_when_read_hook_exists():
    from test_feature_execution_route import _callbacks

    symbols = {
        "example.World": {"kind": "CLASS", "owner": "example.World", "access": 1},
        "readyGetter": {
            "kind": "METHOD", "owner": "example.World", "access": 1,
            "name": "isReady", "parameters": [], "return_type": "boolean",
        },
    }
    choice = select_feature_execution_route([
        {"behavior_id": "native", "event_binding": "test:event",
         "runtime_state_reads": ["ready"], "runtime_state_writes": ["Player health"]}
    ], {"ready": {"binding": "world_state", "verified_symbol_ref": "readyGetter"}},
        symbols, _callbacks())
    assert choice["route"] == "source_owned_java"
    assert choice["verified_behavior_ids"] == ["native"]
    mutation = [row for row in choice["producer_obligations"]
                if row.get("reason") == "runtime_mutation_requires_verified_effect"]
    assert len(mutation) == 1
    assert mutation[0]["name"] == "Player health"

    graph = build_source_work_graph(
        [{"behavior_id": "native", "state_writes": [],
          "runtime_state_writes": ["Player health"]}],
        producer_obligations=choice["producer_obligations"],
        inspected_hooks=_callbacks(),
        verified_behavior_ids=choice["verified_behavior_ids"],
    )
    tasks = [row for row in graph["tasks"] if row["behavior_id"] == "native"]
    native = next(row for row in tasks if row["kind"] == "runtime_mutation")
    assert native["name"] == "Player health"
    assert native["status"] == "pending"
    assert native["evidence"]["runtime_verified"] is False
    effect = next(row for row in tasks if row["kind"] == "gameplay_effect")
    assert native["task_id"] in effect["depends_on"]
    assert not any(row["kind"] == "runtime_producer" and
                   row["name"] == "Player health" for row in tasks)
    scoped = member_source_work(graph, "native", member_role="implement_callback_effects")
    assert any(row["kind"] == "runtime_mutation" for row in scoped["source_tasks"])


def test_exact_checked_world_mutation_is_emitted_with_host_receipt():
    router = Router([{"action": "complete"}, {"verified_call_id": "v0"}])
    output = author_callback_body(
        router, verified_messages(), data_name="native",
        frame=verified_frame(), effect_receipt=True,
    )
    assert "target.apply(hand);" in output
    assert output.index("target.apply(hand);") < output.index("return __mmmHostEffectApplied;")
    selection = router.calls[-1][1]
    assert selection["required_native_state"] == "Player health"
    assert selection["inspected_calls"][0]["signature"] == "example.Target#apply(example.Hand)"
    assert all("getState" not in row["signature"] for row in selection["inspected_calls"])
    assert selection["inspected_calls"][0]["verified_call_id"] == "v0"


def test_no_checked_owner_fails_before_any_small_model_output():
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_RUNTIME_MUTATION_VERIFIED_API_UNAVAILABLE"):
        author_callback_body(
            router, [], data_name="no_source", frame=verified_frame(),
            effect_receipt=True,
        )
    assert router.calls == []


def test_no_semantically_valid_checked_effect_fails_instead_of_substituting():
    router = Router([
        {"action": "complete"}, {"verified_call_id": "unavailable"},
        {"verified_call_id": "unavailable"},
    ])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(
            router, verified_messages(), data_name="missing_mutator",
            frame=verified_frame(), effect_receipt=True,
        )


def test_runtime_mutations_are_not_inferred_from_scalar_state_names():
    assert mutation_targets({"runtime_state_writes": [
        "Player health", "Player health", "Weapon durability",
    ]}) == ("Player health", "Weapon durability")
    with pytest.raises(ValueError, match="CALLBACK_RUNTIME_MUTATION_NAME_INVALID"):
        mutation_targets({"runtime_state_writes": ["Player health", ""]})


def test_nested_model_branch_does_not_get_unconditional_native_mutation():
    router = Router([{"action": "if"}, {"action": "if"}])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(
            router, verified_messages(), data_name="native_guard",
            frame=verified_frame(), effect_receipt=True,
        )
    assert not any("runtime_mutation_Player health" in row[0] for row in router.calls)



def test_native_mutator_with_float_argument_is_not_lost_by_callback_binding():
    # The live player is a verified receiver, but Fabric does not provide the
    # health float in the callback signature. The host freezes setHealth(float)
    # and requests only its missing typed value.
    message = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "parameters": ["example.Target", "example.Hand"],
        },
        "callback_input_api": [{
            "verified_class": "example.Target", "members": [
                {"kind": "METHOD", "name": "setHealth", "static": False,
                 "return_type": "void", "parameters": ["float"]},
                {"kind": "METHOD", "name": "getHealth", "static": False,
                 "return_type": "float", "parameters": []},
            ],
        }],
    })}]
    router = Router([
        {"action": "complete"},
        {"verified_call_id": "m0"},
        {"expression": "6.0f"},
    ])
    source = author_callback_body(
        router, message, data_name="native_float",
        frame=verified_frame(), effect_receipt=True,
    )
    assert "target.setHealth(6.0f);" in source
    assert source.index("target.setHealth(6.0f);") < source.index(
        "return __mmmHostEffectApplied;")
    selection = router.calls[1][1]
    assert selection["inspected_calls"][0]["signature"] == "example.Target#setHealth(float)"
    scalar = router.calls[2][1]
    assert scalar["required_java_type"] == "float"
    assert scalar["required_native_state"] == "Player health"


def test_unbound_non_scalar_parameter_cannot_be_invented_by_small_model():
    message = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "parameters": ["example.Target", "example.Hand"],
        },
        "callback_input_api": [{
            "verified_class": "example.Target", "members": [
                {"kind": "METHOD", "name": "hurt", "static": False,
                 "return_type": "boolean",
                 "parameters": ["example.UnobservedDamageSource", "float"]},
            ],
        }],
    })}]
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_RUNTIME_MUTATION_VERIFIED_API_UNAVAILABLE"):
        author_callback_body(
            router, message, data_name="native_unbound_object",
            frame=verified_frame(), effect_receipt=True,
        )
    assert router.calls == []
