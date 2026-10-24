"""The small coder supplies values; the host writes callback syntax."""
import json

import pytest

from minecraft_mod_ai.host_callback_body import author_callback_body, parse_expression


class OperationRouter:
    def __init__(self, decisions):
        self.decisions = iter(decisions)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append((data_name, parameters, messages))
        return next(self.decisions)


def frame(return_type="void"):
    return {"kind": "method", "prefix": f"private {return_type} apply(Target target) {{\n",
            "suffix": "\n}"}


@pytest.mark.parametrize("bad", [
    "return true;", "true; target.mutate();", "true; } void rogue() {",
    "() -> { return true; }", "new Object() { void rogue() {} }",
    "switch (0) { default -> { target.mutate(); yield true; } }",
])
def test_expression_slot_cannot_contain_statements_or_hidden_members(bad):
    with pytest.raises(ValueError):
        parse_expression(bad)


def test_predicate_return_and_effect_statement_delimiters_are_host_owned():
    predicate = OperationRouter([{"expression": "target.active"}])
    assert "return target.active;" in author_callback_body(
        predicate, [], data_name="predicate", frame=frame("boolean"), predicate=True)
    router = OperationRouter([
        {"action": "call"},
        {"receiver": "target", "method": "mutate", "arguments": ["1", '"a"']},
        {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="effect", frame=frame())
    assert result == frame()["prefix"] + 'target.mutate(1, "a");' + frame()["suffix"]
    first_schema = router.calls[0][1]["properties"]["action"]["enum"]
    assert "return" not in first_schema
    assert "complete" not in first_schema
    assert "complete" in router.calls[2][1]["properties"]["action"]["enum"]


def test_early_return_from_captured_production_cannot_be_an_effect_operation():
    router = OperationRouter([{"action": "return"}, {"action": "return"}])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(router, [], data_name="replay", frame=frame())


def test_host_closes_control_flow_and_names_local_values():
    router = OperationRouter([
        {"action": "declare"}, {"expression": "target.value"},
        {"action": "if"}, {"expression": "mmmValue0 > 0"},
        {"action": "assign"}, {"target": "target.value", "operator": "+=", "value": "mmmValue0"},
        {"action": "else"},
        {"action": "assign"}, {"target": "target.value", "operator": "=", "value": "1"},
        {"action": "end"}, {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="flow", frame=frame())
    assert "var mmmValue0 = target.value;" in result
    assert "if (mmmValue0 > 0) {" in result
    assert "} else {" in result
    assert result.count("{") == result.count("}")


def test_invalid_call_argument_is_corrected_with_rejected_payload_present():
    rejected = {"receiver": "target", "method": "mutate", "arguments": ["1); return;"]}
    router = OperationRouter([
        {"action": "call"}, rejected,
        {"receiver": "target", "method": "mutate", "arguments": ["1"]},
        {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="correction", frame=frame())
    assert "target.mutate(1);" in result
    assert json.loads(router.calls[2][2][-1]["content"])["rejected_payload"] == rejected


def test_predicate_cannot_mutate_canonical_state():
    router = OperationRouter([
        {"expression": "State.writeState_value(1) == null"},
        {"expression": "State.writeState_value(2) == null"},
    ])
    with pytest.raises(ValueError, match="PREDICATE_STATE_WRITE"):
        author_callback_body(router, [], data_name="predicate", frame=frame("boolean"), predicate=True)


def test_host_receipt_marks_only_executed_effects_and_is_not_model_writable():
    router = OperationRouter([
        {"action": "if"}, {"expression": "false"},
        {"action": "assign"}, {"target": "target.value", "operator": "=", "value": "1"},
        {"action": "end"}, {"action": "complete"},
    ])
    result = author_callback_body(router, [], data_name="receipt", frame=frame("boolean"),
                                  effect_receipt=True)
    assert "boolean __mmmHostEffectApplied = false;" in result
    assert "target.value = 1;\n__mmmHostEffectApplied = true;\n}" in result
    assert "return __mmmHostEffectApplied;" in result
    with pytest.raises(ValueError, match="HOST_OWNED"):
        parse_expression("__mmmHostEffectApplied")


def test_model_context_contains_semantic_operations_without_host_receipt_source():
    router = OperationRouter([
        {"action": "assign"}, {"target": "target.value", "operator": "=", "value": "1"},
        {"action": "complete"},
    ])
    source = author_callback_body(router, [], data_name="receipt_boundary", frame=frame("boolean"), effect_receipt=True)
    for _, _, messages in router.calls:
        context = json.loads(messages[-1]["content"])
        assert all("__mmmHostEffectApplied" not in statement for statement in context["accepted_host_statements"])
    assert json.loads(router.calls[0][2][-1]["content"])["accepted_host_statements"] == []
    assert json.loads(router.calls[-1][2][-1]["content"])["accepted_host_statements"] == ["target.value = 1;"]
    assert "boolean __mmmHostEffectApplied = false;" in source
    assert "__mmmHostEffectApplied = true;" in source


def test_only_assigning_a_local_value_cannot_complete_a_gameplay_effect():
    router = OperationRouter([
        {"action": "declare"}, {"expression": "0"},
        {"action": "assign"}, {"target": "mmmValue0", "operator": "=", "value": "1"},
        {"action": "complete"}, {"action": "complete"},
    ])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(router, [], data_name="local_only", frame=frame("boolean"),
                             effect_receipt=True)



def test_event_callback_registry_cannot_be_invoked_by_condition():
    # Captured from a failed Minecraft/Fabric compiler run: a listener's
    # predicate must not recursively call its own event dispatcher.
    bad = ("net.fabricmc.fabric.api.event.player.UseItemCallback.EVENT"
           ".interact(player, level, interactionHand)"
           " == net.minecraft.world.InteractionResult.ResultType.PASS")
    with pytest.raises(ValueError, match="CALLBACK_PREDICATE_EVENT_DISPATCH"):
        parse_expression(bad, predicate=True)


def test_event_registry_mistake_is_rejected_and_an_actual_condition_is_selected():
    bad = "example.UseItemCallback.EVENT.interact(player, level, hand) == PASS"
    router = OperationRouter([
        {"expression": bad},
        {"expression": "target.active"},
    ])
    result = author_callback_body(
        router, [], data_name="event_condition",
        frame=frame("boolean"), predicate=True,
    )
    assert "return target.active;" in result
    assert "EVENT" not in result
    correction = json.loads(router.calls[1][2][-1]["content"])
    assert "CALLBACK_PREDICATE_EVENT_DISPATCH" in correction["last_error"]


def test_compiler_failed_predicate_cannot_be_reaccepted_unchanged():
    original = frame("boolean")["prefix"] + "return target.active;" + frame()["suffix"]
    router = OperationRouter([
        {"expression": "target.active"},
        {"expression": "target.ready"},
    ])
    revised = author_callback_body(
        router, [], data_name="repair_predicate", frame=frame("boolean"),
        predicate=True, rejected_compiler_member=original,
        compiler_diagnostic="Demo.java:4: error: cannot find symbol target.active",
    )
    assert "target.ready" in revised
    assert revised != original
    retry = json.loads(router.calls[1][2][-1]["content"])
    assert "CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED" in retry["last_error"]
    assert "cannot find symbol" in retry["javac_diagnostic"]
    assert retry["javac_rejected_member"] == original


def test_compiler_failed_effect_cannot_be_reaccepted_unchanged():
    original = frame()["prefix"] + "target.mutate(1);" + frame()["suffix"]
    router = OperationRouter([
        {"action": "call"},
        {"receiver": "target", "method": "mutate", "arguments": ["1"]},
        {"action": "complete"},
    ])
    with pytest.raises(ValueError, match="CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED"):
        author_callback_body(
            router, [], data_name="repair_effect", frame=frame(),
            rejected_compiler_member=original,
            compiler_diagnostic="Demo.java:4: error: cannot find symbol mutate(int)",
        )



def test_compiler_rejected_predicate_with_whitespace_changes_is_still_rejected():
    old = frame("boolean")["prefix"] + "return target.active;" + frame()["suffix"]
    router = OperationRouter([
        {"expression": "target . active"},
        {"expression": "target.ready"},
    ])
    result = author_callback_body(
        router, [], data_name="repair_tokens", frame=frame("boolean"),
        predicate=True, rejected_compiler_member=old,
        compiler_diagnostic="Demo.java:1: error: cannot find symbol active",
    )
    assert "target.ready" in result
    assert len(router.calls) == 2
    retry = json.loads(router.calls[1][2][-1]["content"])
    assert "CALLBACK_COMPILER_REJECTED_MEMBER_REPEATED" in retry["last_error"]



def _verified_messages():
    return [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "parameters": ["example.Target", "example.Hand"],
        },
        "callback_input_api": [{
            "verified_class": "example.Target", "members": [
                {"kind": "METHOD", "name": "isReady", "static": False,
                 "return_type": "boolean", "parameters": []},
                {"kind": "METHOD", "name": "apply", "static": False,
                 "return_type": "void", "parameters": ["example.Hand"]},
                {"kind": "METHOD", "name": "madeUpCall", "static": True,
                 "return_type": "void", "parameters": []},
                {"kind": "METHOD", "name": "unsupported", "static": False,
                 "return_type": "void", "parameters": ["example.Missing"]},
            ],
        }],
    })}]


def _verified_frame(return_type="void"):
    return {"kind": "method",
            "prefix": ("private " + return_type
                       + " execute(example.Target target, example.Hand hand) {\n"),
            "suffix": "\n}"}


def test_verified_callback_method_choice_compiles_from_indexed_abi():
    router = OperationRouter([
        {"action": "call"}, {"verified_call_id": "v0"}, {"action": "complete"},
    ])
    result = author_callback_body(router, _verified_messages(),
                                  data_name="verified_effect", frame=_verified_frame())
    assert "target.apply(hand);" in result
    options = json.loads(router.calls[1][2][-1]["content"])["verified_candidates"]
    assert options[0]["expression"] == "target.apply(hand)"
    assert all("madeUpCall" not in x["expression"] and
               "unsupported" not in x["expression"] for x in options)
    assert "method" not in router.calls[1][1]["properties"]
    assert "receiver" not in router.calls[1][1]["properties"]


def test_verified_callback_predicate_choice_is_not_a_java_expression_field():
    router = OperationRouter([{"verified_call_id": "v0"}])
    result = author_callback_body(router, _verified_messages(),
                                  data_name="verified_pred", frame=_verified_frame("boolean"),
                                  predicate=True)
    assert "return target.isReady();" in result
    assert list(router.calls[0][1]["properties"]) == ["verified_call_id"]
    assert router.calls[0][1]["properties"]["verified_call_id"]["enum"] == ["v0"]


def test_verified_callback_candidate_requires_real_declared_input_types():
    from minecraft_mod_ai.host_verified_callback_ops import verified_callback_choices
    matches = verified_callback_choices(_verified_messages(), _verified_frame())
    assert {x["expression"] for x in matches} == {"target.apply(hand)"}
    wrong_frame = {"kind": "method",
                   "prefix": "private void apply(example.Other other) {\n",
                   "suffix": "\n}"}
    assert verified_callback_choices(_verified_messages(), wrong_frame) == []



def test_callback_api_retrieval_uses_full_index_not_alphabetical_first_page():
    from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
    catalog = FileAPIEvidence.__new__(FileAPIEvidence)
    catalog.classes = {"example.Target": "example.Target"}
    catalog.symbols = {"example.Target": {
        "kind": "CLASS", "name": "Target", "owner": "example.Target", "access": 1,
    }}
    catalog._accessible = lambda owner: True
    options = []
    for idx in range(80):
        options.append((str(idx), {
            "kind": "METHOD", "owner": "example.Target",
            "name": "arbitrary" + str(idx), "return_type": "void",
            "static": False, "parameters": [],
        }))
    options.append(("open", {
        "kind": "METHOD", "owner": "example.Target",
        "name": "openTradingMenu", "return_type": "void",
        "static": False, "parameters": [],
    }))
    options.append(("invalid", {
        "kind": "METHOD", "owner": "example.Target",
        "name": "openTradingMenuWithUnknownArgument",
        "return_type": "void", "static": False,
        "parameters": ["example.ItemStack"],
    }))
    catalog._member_options = lambda owner: options
    api = catalog.callback_member_records(
        ["example.Target"], intent="Player opens trading menu")
    assert len(api) == 1
    names = [row["name"] for row in api[0]["members"]]
    assert names == ["openTradingMenu"]
    assert api[0]["member_count"] == len(options)
    assert api[0]["members_truncated"] is True



def test_model_typed_local_declaration_is_lowered_to_host_owned_initializer():
    # Production failed when the coder repeated "boolean mmmValue0 = false;"
    # while the host asked for one expression. Recover the parsed initializer.
    router = OperationRouter([
        {"action": "declare"}, {"expression": "boolean mmmValue0 = false;"},
        {"action": "assign"},
        {"target": "target.active", "operator": "=", "value": "mmmValue0"},
        {"action": "complete"},
    ])
    result = author_callback_body(
        router, [], data_name="typed_initializer", frame=frame("boolean"),
        effect_receipt=True)
    assert "var mmmValue0 = false;" in result
    assert "target.active = mmmValue0;" in result
    assert "boolean mmmValue0 = false;" not in result
    assert "__mmmHostEffectApplied = true;" in result


@pytest.mark.parametrize("bad", [
    "boolean mmmValue0 = false; target.mutate();",
    "boolean mmmValue0 = false, injected = true;",
    "boolean wrongName = false;",
    "boolean mmmValue0;",
    "return false;",
])
def test_typed_initializer_cannot_smuggle_extra_statements_or_symbols(bad):
    from minecraft_mod_ai.host_callback_body import parse_initializer
    with pytest.raises(ValueError):
        parse_initializer(bad, host_local="mmmValue0")



def test_callback_abi_retains_baseline_getter_and_return_constant_without_prompt_overlap():
    from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
    api = FileAPIEvidence.__new__(FileAPIEvidence)
    api.classes = {"example.Player": "example.Player", "example.Result": "example.Result"}
    api.symbols = {owner: {"kind": "CLASS", "owner": owner, "access": 1,
                           "name": owner.split(".")[-1]} for owner in api.classes}
    api._accessible = lambda owner: True
    options = {
        "example.Player": [("uuid", {
            "kind": "METHOD", "owner": "example.Player",
            "name": "getUUID", "access": 1,
            "return_type": "java.util.UUID", "parameters": [],
        })],
        "example.Result": [("pass", {
            "kind": "FIELD", "owner": "example.Result",
            "name": "PASS", "return_type": "example.Result", "access": 25,
        })],
    }
    api._member_options = lambda owner: options[owner]
    # Neither method name is mentioned by the behavior specification. The
    # host must still retain typed callback inputs and return-type constants.
    records = api.callback_member_records(
        list(api.classes), intent="Assemble floating workshop", predicate=True)
    by_owner = {row["verified_class"]: row for row in records}
    assert set(by_owner) == set(api.classes)
    assert [row["name"] for row in by_owner["example.Player"]["members"]] == ["getUUID"]
    assert [row["name"] for row in by_owner["example.Result"]["members"]] == ["PASS"]



def test_callback_api_cap_does_not_starve_later_input_owner():
    from minecraft_mod_ai.construction_file_api_evidence import FileAPIEvidence
    a, b = "example.A", "example.B"
    api = FileAPIEvidence.__new__(FileAPIEvidence)
    api.classes = {a: a, b: b}
    api.symbols = {name: {"kind": "CLASS", "owner": name,
                          "access": 1, "name": name.rsplit(".", 1)[-1]}
                   for name in (a, b)}
    api._accessible = lambda _name: True
    options = {a: [
        (str(i), {"kind": "METHOD", "owner": a,
                  "name": "getItem" + str(i), "access": 1,
                  "return_type": "int", "parameters": []})
        for i in range(100)],
        b: [("unique", {"kind": "METHOD", "owner": b,
                        "name": "getWorldTime", "access": 1,
                        "return_type": "long", "parameters": []})],
    }
    api._member_options = lambda name: options[name]
    result = api.callback_member_records([a, b], intent="No matching operation",
                                         limit=2)
    assert len(result) == 2
    assert len(result[0]["members"]) == 1
    assert [x["name"] for x in result[1]["members"]] == ["getWorldTime"]


def test_repeated_model_authored_receipt_switches_to_a_real_effect_operation():
    # Captured production shape: the small coder repeatedly tried to declare
    # the receipt variable reserved to the host instead of implementing an
    # actual game state change. A safe recovery must not accept the receipt as
    # evidence or trap the coder in the same invalid 'declare' slot.
    router = OperationRouter([
        {"action": "declare"},
        {"expression": "boolean __mmmHostEffectApplied = false;"},
        {"expression": "boolean __mmmHostEffectApplied = false;"},
        {"action": "assign"},
        {"target": "target.value", "operator": "=", "value": "4"},
        {"action": "complete"},
    ])
    source = author_callback_body(
        router, [], data_name="receipt_recovery", frame=frame("boolean"),
        effect_receipt=True,
    )
    assert source.count("boolean __mmmHostEffectApplied = false;") == 1
    assert "target.value = 4;" in source
    assert "__mmmHostEffectApplied = true;" in source
    assert "declare" not in router.calls[3][1]["properties"]["action"]["enum"]
    assert json.loads(router.calls[3][2][-1]["content"])[
        "blocked_non_effect_operations"] == ["declare"]


def test_inspected_callback_never_offers_unverified_receiver_method_pair():
    """Log 34 attempted to fabricate ItemEvents.register(UseOnCallback, ...)."""
    message = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "event_owner": "net.fabricmc.fabric.api.event.player.FakeItemEvents",
            "parameters": ["example.Target"],
            "return_type": "void",
        },
        # No signed callable mutation from that receiver.
        "callback_input_api": [{
            "verified_class": "example.Target",
            "members": [{"kind": "METHOD", "name": "getValue",
                         "static": False, "return_type": "int",
                         "parameters": []}],
        }],
    })}]
    class Capture:
        schema = None
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            self.schema = parameters
            raise RuntimeError("captured the host-owned allowed operation enum")
    router = Capture()
    with pytest.raises(RuntimeError, match="captured"):
        author_callback_body(router, message, data_name="runtime_effect",
                             frame={"kind": "method",
                                    "prefix": "private void process(example.Target target) {\n",
                                    "suffix": "\n}"})
    assert "call" not in router.schema["properties"]["action"]["enum"]


def test_verified_fabric_event_registration_is_not_gameplay_effect():
    from minecraft_mod_ai.host_verified_callback_ops import verified_callback_choices

    evidence = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {"parameters": [
            "net.fabricmc.fabric.api.event.Event"
        ]},
        "callback_input_api": [{
            "verified_class": "net.fabricmc.fabric.api.event.Event",
            "members": [{"kind": "METHOD", "name": "register",
                         "static": False, "return_type": "void",
                         "parameters": []}],
        }],
    })}]
    frame = {"kind": "method",
             "prefix": "private void run(net.fabricmc.fabric.api.event.Event evt) {\n",
             "suffix": "\n}"}
    assert verified_callback_choices(evidence, frame) == []


def test_missing_runtime_producer_requires_real_receiver_observation():
    from minecraft_mod_ai.host_callback_body import (
        validate_runtime_observation_predicate,
    )

    payload = {
        "source_owned_work": {"source_tasks": [{
            "kind": "runtime_producer", "name": "Available planets",
            "status": "pending",
        }]},
        "observed_source_name": "Available planets",
        "declaration_metadata": {
            "parameters": [{"name": "player", "type": "example.Player"}],
        },
    }
    scoped = [{"role": "user", "content": json.dumps(payload)}]
    with pytest.raises(ValueError, match="SOURCE_RUNTIME_PRODUCER_NOT_IMPLEMENTED"):
        validate_runtime_observation_predicate(scoped, "player != null")
    with pytest.raises(ValueError, match="SOURCE_RUNTIME_OBSERVATION_NOT_GROUNDED"):
        validate_runtime_observation_predicate(scoped, "true")
    validate_runtime_observation_predicate(scoped, "player.level() != null")


def test_feature_completion_requires_its_real_host_owned_state_write():
    from minecraft_mod_ai.host_callback_body import (
        _require_canonical_state_effects, _effect_operation,
    )

    frame = {
        "host_behavior_contract": {
            "behavior_id": "colonize", "state_writes": ["colonized"],
        },
        "host_state_access": [
            {"owner": "example.AuthoredStateModel",
             "methods": [{"name": "writeState_colonized"}]},
        ],
    }
    # Local computations are NOT side effects, even if the initializer calls
    # a real method. A status/log only or other state's write is not enough.
    assert not _effect_operation("declare", "var mmmValue0 = player.getLevel();")
    with pytest.raises(ValueError, match="CANONICAL_STATE_WRITES_MISSING"):
        _require_canonical_state_effects(
            "private void apply() { var mmmValue0 = player.getLevel(); }", frame,
        )
    with pytest.raises(ValueError, match="CANONICAL_STATE_WRITES_MISSING"):
        _require_canonical_state_effects(
            "private void apply() {"
            " example.AuthoredStateModel.writeState_credits(1); }", frame,
        )
    _require_canonical_state_effects(
        "private void apply() {"
        " example.AuthoredStateModel.writeState_colonized(true); }", frame,
    )


def test_unmaterialized_canonical_state_owner_cannot_be_claimed():
    from minecraft_mod_ai.host_callback_body import _require_canonical_state_effects

    frame = {"host_behavior_contract": {"state_writes": ["colonized"]},
             "host_state_access": []}
    with pytest.raises(ValueError, match="CALLBACK_STATE_OWNER_UNAVAILABLE"):
        _require_canonical_state_effects(
            "private void apply() { System.out.println(\"colonized\"); }",
            frame,
        )
