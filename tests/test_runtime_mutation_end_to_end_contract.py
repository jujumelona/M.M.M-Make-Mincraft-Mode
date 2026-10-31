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
                {"kind": "METHOD", "name": "setHealth", "static": False,
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
    router = Router([{"verified_call_id": "v1"}])
    output = author_callback_body(
        router, verified_messages(), data_name="native",
        frame=verified_frame(), effect_receipt=True,
    )
    assert "target.setHealth(hand);" in output
    assert output.index("target.setHealth(hand);") < output.index("return __mmmHostEffectApplied;")
    selection = router.calls[-1][1]
    assert selection["required_native_state"] == "Player health"
    assert selection["inspected_calls"][0]["signature"] == "example.Target#setHealth(example.Hand)"
    assert all("getState" not in row["signature"] for row in selection["inspected_calls"])
    assert selection["inspected_calls"][0]["verified_call_id"] == "v1"


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
    selection = router.calls[0][1]
    assert selection["inspected_calls"][0]["signature"] == "example.Target#setHealth(float)"
    scalar = router.calls[1][1]
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



def test_compiler_repair_cannot_drop_a_checked_native_mutation():
    import hashlib

    from minecraft_mod_ai.runtime_mutation_author import (
        verify_native_runtime_mutation_coverage,
    )
    identity = "native-proof"
    method = "applyBehavior" + hashlib.sha256(
        identity.encode("utf-8")).hexdigest()[:20]
    behavior = [{"behavior_id": identity, "runtime_state_writes": ["Player health"]}]
    receipt = [{
        "behavior_id": identity, "state_name": "Player health",
        "verified_signature": "example.Target#setHealth(float)",
        "statement": "target.setHealth(6.0f);",
    }]
    source = (
        "package example; class Bridge { private boolean " + method +
        "(Target target) { target.setHealth(6.0f); return true; } }"
    )
    verify_native_runtime_mutation_coverage(source, behavior, receipt)
    # This source still parses and can compile when Target has both methods,
    # but it no longer implements the immutable state transition.
    with pytest.raises(ValueError, match="NATIVE_MUTATION_PROOF_EFFECT_DROPPED"):
        verify_native_runtime_mutation_coverage(
            source.replace("target.setHealth(6.0f);", "target.getHealth();"),
            behavior, receipt,
        )
    with pytest.raises(ValueError, match="NATIVE_MUTATION_PROOF_EFFECT_DROPPED"):
        verify_native_runtime_mutation_coverage(
            source.replace(
                "target.setHealth(6.0f);",
                "if (false) { target.setHealth(6.0f); }",
            ),
            behavior, receipt,
        )
    with pytest.raises(ValueError, match="NATIVE_MUTATION_PROOF_COVERAGE_MISSING"):
        verify_native_runtime_mutation_coverage(source, behavior, [])
    with pytest.raises(ValueError, match="NATIVE_MUTATION_PROOF_HANDLER_MISSING"):
        verify_native_runtime_mutation_coverage(
            source.replace(method, "otherMethod"), behavior, receipt,
        )


def test_unrelated_checked_callable_cannot_claim_native_effect():
    # Indexed calls such as apply() may be valid Java but cannot certify an
    # unrelated gameplay state change.
    payload = json.loads(verified_messages()[0]["content"])
    payload["callback_input_api"][0]["members"] = [
        member for member in payload["callback_input_api"][0]["members"]
        if member["name"] != "setHealth"
    ]
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_RUNTIME_MUTATION_BACKING_API_UNRESOLVED"):
        author_callback_body(
            router, [{"role": "user", "content": json.dumps(payload)}],
            data_name="irrelevant_mutation", frame=verified_frame(),
            effect_receipt=True,
        )
    assert not any("_runtime_mutation_" in name for name, _, _ in router.calls)



def test_native_mutation_candidates_reject_telemetry_even_if_name_matches():
    from minecraft_mod_ai.runtime_mutation_author import (
        _native_state_effect_method, _state_backed_native_candidates,
    )
    methods = [
        ("t0", {"name": "recordDamage", "signature": "example.CombatTracker#recordDamage(float)"}),
        ("t1", {"name": "broadcastDamageEvent", "signature": "example.World#broadcastDamageEvent(float)"}),
        ("t2", {"name": "showVehicleHealth", "signature": "example.Entity#showVehicleHealth()"}),
        ("m3", {"name": "setHealth", "signature": "example.Player#setHealth(float)"}),
    ]
    damage = _state_backed_native_candidates("Combat damage", methods)
    assert {m["name"] for _, m in damage} == {"recordDamage", "broadcastDamageEvent"}
    assert not any(_native_state_effect_method(m["name"]) for _, m in damage)
    assert _native_state_effect_method("setHealth")
    assert not _native_state_effect_method("showVehicleHealth")


def test_native_player_health_cannot_target_arbitrary_world_player_getter():
    from minecraft_mod_ai.runtime_mutation_author import _state_owner_receiver_admissible

    frame = {"prefix": (
        "private boolean apply(net.minecraft.server.level.ServerLevel level, "
        "net.minecraft.world.entity.LivingEntity victim) {\n"
    )}
    unsafe = [
        "level.getRandomPlayer()",
        "victim.getLastHurtByPlayer()",
        "level.getPlayers().get(0)",
    ]
    for receiver in unsafe:
        candidate = {
            "receiver": receiver,
            "signature": "net.minecraft.server.level.ServerPlayer#setHealth(float)",
        }
        assert not _state_owner_receiver_admissible("Player health", candidate, frame)
    assert not _state_owner_receiver_admissible("Player health", {
        "receiver": "victim",
        "signature": "net.minecraft.world.entity.LivingEntity#setHealth(float)",
    }, frame)


def test_player_native_health_mutator_must_bind_to_event_player():
    from minecraft_mod_ai.runtime_mutation_author import _state_owner_receiver_admissible

    frame = {"prefix": (
        "private boolean apply(net.minecraft.server.level.ServerPlayer player, "
        "net.minecraft.server.level.ServerLevel level) {\n"
    )}
    direct = {"receiver": "player",
              "signature": "net.minecraft.server.level.ServerPlayer#setHealth(float)"}
    assert _state_owner_receiver_admissible("Player health", direct, frame)
    assert not _state_owner_receiver_admissible("Player health", {
        **direct, "receiver": "level.getRandomPlayer()",
    }, frame)
    assert not _state_owner_receiver_admissible("Player health", {
        **direct, "receiver": "level",
    }, frame)



def test_runtime_mutation_author_rejects_damage_log_and_arbitrary_player():
    from minecraft_mod_ai.runtime_mutation_author import author_runtime_mutation_calls

    frame = {
        "prefix": (
            "private boolean apply(net.minecraft.server.level.ServerLevel serverLevel, "
            "net.minecraft.world.entity.LivingEntity victim) {\n"),
        "suffix": "\n}",
        "host_behavior_contract": {"behavior_id": "combat"},
    }
    telemetry = [{
        "id": "m0",
        "name": "recordDamage",
        "receiver": "victim.getCombatTracker()",
        "signature": "net.minecraft.world.damagesource.CombatTracker#recordDamage(float)",
        "arguments": [{"type": "float", "expression": None}],
    }]
    with pytest.raises(ValueError, match="CALLBACK_RUNTIME_MUTATION_BACKING_API_UNRESOLVED"):
        author_runtime_mutation_calls(
            Router([]), [], data_name="record_only", frame=frame,
            verified_calls=[], typed_candidates=telemetry,
            targets=["Combat damage"], statements=[],
            semantic_statements=[], effect_receipt=True,
            receipt_name="__mmmHostEffectApplied",
        )
    random_player = [{
        "id": "m0",
        "name": "setHealth",
        "receiver": "serverLevel.getRandomPlayer()",
        "signature": "net.minecraft.server.level.ServerPlayer#setHealth(float)",
        "arguments": [{"type": "float", "expression": None}],
    }]
    with pytest.raises(ValueError, match="CALLBACK_RUNTIME_MUTATION_BACKING_API_UNRESOLVED"):
        author_runtime_mutation_calls(
            Router([]), [], data_name="unowned_player", frame=frame,
            verified_calls=[], typed_candidates=random_player,
            targets=["Player health"], statements=[],
            semantic_statements=[], effect_receipt=True,
            receipt_name="__mmmHostEffectApplied",
        )



def _verified_item_durability_payload(*, include_max_damage=True):
    methods = [
        {"kind": "METHOD", "name": "setDamageValue", "static": False,
         "return_type": "void", "parameters": ["int"]},
        {"kind": "METHOD", "name": "getDamageValue", "static": False,
         "return_type": "int", "parameters": []},
    ]
    if include_max_damage:
        methods.append({"kind": "METHOD", "name": "getMaxDamage",
                        "static": False, "return_type": "int", "parameters": []})
    item_type = "net.minecraft.world.item.ItemStack"
    return [{
        "role": "user",
        "content": json.dumps({
            "verified_event_callback": {"parameters": [item_type]},
            "declaration_metadata": {"parameters": [
                {"name": "stack", "type": item_type}]},
            "callback_input_api": [{
                "verified_class": item_type, "members": methods,
            }],
        }),
    }]


def _durability_frame():
    return {
        "kind": "method",
        "prefix": ("private boolean apply("
                   "net.minecraft.world.item.ItemStack stack) {\n"),
        "suffix": "\n}",
        "host_behavior_contract": {
            "behavior_id": "generic-wear",
            "runtime_state_writes": ["Weapon durability"],
            "state_reads": [], "state_writes": [],
        },
    }


def test_verified_item_damage_trait_compiles_incremental_durability_effect():
    router = Router([{"verified_call_id": "m0"}, {"expression": "2"}])
    source = author_callback_body(
        router, _verified_item_durability_payload(),
        data_name="item_wear", frame=_durability_frame(), effect_receipt=True,
    )
    assert "stack.setDamageValue(" in source
    assert "stack.getDamageValue() + Math.max(1L, (long) (2))" in source
    assert "stack.getMaxDamage()" in source
    assert router.calls[0][1]["inspected_calls"][0]["signature"] == (
        "net.minecraft.world.item.ItemStack#setDamageValue(int)")


def test_unverified_item_damage_trait_does_not_guess_durability_mutation():
    router = Router([])
    with pytest.raises(
            ValueError, match="CALLBACK_RUNTIME_MUTATION_BACKING_API_UNRESOLVED"):
        author_callback_body(
            router, _verified_item_durability_payload(include_max_damage=False),
            data_name="item_wear", frame=_durability_frame(), effect_receipt=True)
    assert router.calls == []


def test_verified_durability_java_is_executable_with_real_increment_not_reset(tmp_path):
    import shutil
    import subprocess

    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK compiler and runtime are required")
    source = author_callback_body(
        Router([{"verified_call_id": "m0"}, {"expression": "2"}]),
        _verified_item_durability_payload(), data_name="wear_jvm",
        frame=_durability_frame(), effect_receipt=True,
    )
    item = """
package net.minecraft.world.item;
public final class ItemStack {
    int damage;
    public ItemStack(int value) { damage = value; }
    public int getDamageValue() { return damage; }
    public int getMaxDamage() { return 20; }
    public void setDamageValue(int value) { damage = value; }
}
"""
    main = """
package example;
public class Bridge {
""" + source + """
    public static void main(String[] args) {
        Bridge bridge = new Bridge();
        var stack = new net.minecraft.world.item.ItemStack(5);
        if (!bridge.apply(stack) || stack.getDamageValue() != 7)
            throw new AssertionError("must increment from existing damage");
        if (!bridge.apply(stack) || stack.getDamageValue() != 9)
            throw new AssertionError("must accumulate damage");
        System.out.println("ITEM_DAMAGE_DELTA_PASS");
    }
}
"""
    paths = []
    for filename, contents in {
        "example/Bridge.java": main,
        "net/minecraft/world/item/ItemStack.java": item,
    }.items():
        file = tmp_path / filename
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(contents, encoding="utf-8")
        paths.append(str(file))
    compiled = subprocess.run(
        [javac, "-proc:none", "-d", str(tmp_path), *paths],
        capture_output=True, text=True, timeout=60, check=False)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [java, "-cp", str(tmp_path), "example.Bridge"],
        capture_output=True, text=True, timeout=30, check=False)
    assert executed.returncode == 0, executed.stderr
    assert "ITEM_DAMAGE_DELTA_PASS" in executed.stdout
