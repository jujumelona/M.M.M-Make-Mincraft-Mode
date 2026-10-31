"""Regression: required canonical state transitions are host-owned compiler work."""

import pytest

from minecraft_mod_ai.host_callback_body import author_callback_body


class Router:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.calls = []

    def generate_structured_data(self, role, messages, *, data_name, parameters):
        self.calls.append(data_name)
        return next(self.rows)


def effect_frame(*, declared=("credits",), required=("credits",)):
    methods = [
        {"name": "writeState_" + name,
         "signature": "public static void writeState_" + name + "(long value) { }"}
        for name in declared
    ]
    return {
        "kind": "method",
        "prefix": "private boolean apply(Target target) {\n",
        "suffix": "\n}",
        "host_state_access": [{"owner": "example.AuthoredStateModel", "methods": methods}],
        "host_behavior_contract": {
            "behavior_id": "trade", "intent": "purchase a ship upgrade",
            "state_reads": ["credits"], "state_writes": list(required),
        },
    }


def test_required_state_write_is_lowered_with_exact_owner_and_before_receipt_return():
    router = Router([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "19L"},
    ])
    source = author_callback_body(
        router, [], data_name="ship_purchase", frame=effect_frame(),
        effect_receipt=True,
    )
    writer = "example.AuthoredStateModel.writeState_credits(19L);"
    assert writer in source
    assert source.index(writer) < source.index("return __mmmHostEffectApplied;")
    assert router.calls[-1] == "ship_purchase_required_state_credits_try_0"


def test_missing_declared_state_is_rejected_before_any_model_work():
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED: credits"):
        author_callback_body(router, [], data_name="unowned", frame=effect_frame(declared=()),
                             effect_receipt=True)
    assert router.calls == []


def test_write_into_different_state_cannot_satisfy_original_obligation():
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED: credits"):
        author_callback_body(router, [], data_name="wrong_state",
                             frame=effect_frame(declared=("ship_level",)),
                             effect_receipt=True)
    assert router.calls == []


def test_model_does_not_choose_writable_state_identifier():
    router = Router([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "21L"},
        {"expression": "22L"},
    ])
    source = author_callback_body(
        router, [], data_name="two_state", frame=effect_frame(
            declared=("credits", "ship_level"), required=("credits", "ship_level")),
        effect_receipt=True,
    )
    assert "example.AuthoredStateModel.writeState_credits(21L);" in source
    assert "example.AuthoredStateModel.writeState_ship_level(22L);" in source
    assert router.calls[-2:] == [
        "two_state_required_state_credits_try_0", "two_state_required_state_ship_level_try_0",
    ]



def test_late_required_state_writer_never_bypasses_an_effect_condition():
    # Earlier host lowering appended the required mutation after the closing
    # brace, so an effect hidden behind if(false) still changed canonical
    # state. Reject the semantic ownership mismatch rather than silently
    # converting a conditional effect into an unconditional one.
    router = Router([{"action": "if"}, {"action": "if"}])
    with pytest.raises(ValueError, match="CALLBACK_OPERATION_INVALID"):
        author_callback_body(
            router, [], data_name="conditional_purchase",
            frame=effect_frame(), effect_receipt=True,
        )
    assert not any("required_state_credits" in name for name in router.calls)



def test_required_writer_ignores_context_overload_and_exposes_exact_value_type():
    from minecraft_mod_ai.host_callback_body import _required_state_write_slots
    frame = effect_frame()
    frame["host_state_access"][0]["methods"].insert(0, {
        "name": "writeState_credits",
        "signature": "public static void writeState_credits(long value, java.util.Map<String, Object> context) { }",
    })
    slot = _required_state_write_slots(frame)["credits"]
    assert slot["value_type"] == "long"
    assert "(long value)" in slot["signature"]
    router = Router([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "25L"},
    ])
    author_callback_body(router, [], data_name="typed_write", frame=frame,
                         effect_receipt=True)
    request = None
    # Router records actions by name. A separate capture router can verify
    # exact contextual Java type in the mandatory writer request.
    class CaptureRouter(Router):
        def generate_structured_data(self, role, messages, *, data_name, parameters):
            if "required_state_credits" in data_name:
                nonlocal request
                request = json.loads(messages[-1]["content"])
            return super().generate_structured_data(
                role, messages, data_name=data_name, parameters=parameters)
    import json
    capture = CaptureRouter([
        {"action": "assign"},
        {"target": "target.amount", "operator": "=", "value": "5"},
        {"action": "complete"},
        {"expression": "25L"},
    ])
    author_callback_body(capture, [], data_name="typed_write", frame=frame,
                         effect_receipt=True)
    assert request["required_java_value_type"] == "long"


def test_required_state_write_rejects_context_only_overload():
    from minecraft_mod_ai.host_callback_body import _required_state_write_slots
    frame = effect_frame()
    frame["host_state_access"][0]["methods"] = [{
        "name": "writeState_credits",
        "signature": "public static void writeState_credits(long value, java.util.Map<String, Object> context) { }",
    }]
    with pytest.raises(ValueError, match="CALLBACK_REQUIRED_STATE_ACCESSOR_UNDECLARED"):
        _required_state_write_slots(frame)



def test_real_prerequisite_header_without_method_body_is_accepted():
    # java_api_declarations strips bodies and persists a semicolon signature,
    # not a hand-written full Java method as in most synthetic tests.
    from minecraft_mod_ai.host_callback_body import _required_state_write_slots
    frame = effect_frame()
    frame["host_state_access"][0]["methods"] = [
        {"name": "writeState_credits",
         "signature": "public static synchronized void writeState_credits(long value);"},
        {"name": "writeState_credits",
         "signature": "public static synchronized void writeState_credits(long value, java.util.Map<String,Object> context);"},
    ]
    slots = _required_state_write_slots(frame)
    assert slots["credits"]["value_type"] == "long"
    assert slots["credits"]["signature"].endswith("(long value);")


def test_player_scoped_write_uses_callback_uuid_not_global_state():
    frame = effect_frame()
    frame["prefix"] = (
        "private boolean apply(net.minecraft.server.network.ServerPlayerEntity player) {\n"
    )
    frame["host_state_access"][0]["state_scopes"] = {"credits": "player"}
    router = Router([
        {"action": "complete"},
        {"expression": "19L"},
    ])
    source = author_callback_body(
        router, [], data_name="player_purchase", frame=frame,
        effect_receipt=True,
    )
    assert ('example.AuthoredStateModel.writeState_credits('
            '19L, java.util.Map.of("player", player.getUuid()));') in source
    assert 'example.AuthoredStateModel.writeState_credits(19L);' not in source


def test_player_scoped_write_fails_before_model_if_event_has_no_player():
    frame = effect_frame()
    frame["host_state_access"][0]["state_scopes"] = {"credits": "player"}
    router = Router([])
    with pytest.raises(ValueError, match="CALLBACK_PLAYER_STATE_OWNER_UNRESOLVED"):
        author_callback_body(
            router, [], data_name="missing_player", frame=frame,
            effect_receipt=True,
        )
    assert router.calls == []


def test_player_scoped_write_supports_mojang_named_player_api():
    frame = effect_frame()
    frame["prefix"] = (
        "private boolean apply(net.minecraft.server.level.ServerPlayer owner) {\n"
    )
    frame["host_state_access"][0]["state_scopes"] = {"credits": "player"}
    router = Router([
        {"action": "complete"},
        {"expression": "19L"},
    ])
    source = author_callback_body(
        router, [], data_name="mojang_player", frame=frame,
        effect_receipt=True,
    )
    assert 'java.util.Map.of("player", owner.getUUID())' in source


def test_compiled_player_state_delta_is_atomic_and_isolated_per_uuid(tmp_path):
    """Real JVM probe: generated callback must not share or reset player money."""
    import json
    import shutil
    import subprocess

    from minecraft_mod_ai.host_prerequisite_api import java_api_declarations
    from minecraft_mod_ai.host_typed_state_access import state_access_contract
    from minecraft_mod_ai.production_state_compiler import render_production_state_java
    from test_production_state_compiler import _state_section

    javac, java = shutil.which("javac"), shutil.which("java")
    if not javac or not java:
        pytest.skip("JDK compiler/runtime required")

    section = _state_section(variables=[{
        "name": "credits", "owner": "player", "type": "long",
        "unit": "credits", "default": "0", "domain": "nonnegative",
    }])
    state = render_production_state_java(section, package_name="proof")
    api = java_api_declarations(
        state, path="src/main/java/proof/AuthoredStateModel.java")
    contracts = state_access_contract([api])
    assert contracts[0]["state_scopes"] == {"credits": "player"}

    frame = {
        "kind": "method",
        "prefix": (
            "private boolean apply("
            "net.minecraft.server.network.ServerPlayerEntity player) {\n"
        ),
        "suffix": "\n}",
        "host_state_access": contracts,
        "host_behavior_contract": {
            "behavior_id": "earn_credits",
            "state_reads": ["credits"],
            "state_writes": ["credits"],
        },
    }
    msg = [{"role": "user", "content": json.dumps({
        "verified_event_callback": {
            "parameters": ["net.minecraft.server.network.ServerPlayerEntity"]
        },
        "callback_input_api": [],
    })}]
    callback = author_callback_body(
        Router([{"value": 5, "operator": "add"}]),
        msg, data_name="scoped_increment", frame=frame, effect_receipt=True,
    )
    assert "synchronized (proof.AuthoredStateModel.class)" in callback
    assert 'java.util.Map.of("player", player.getUuid())' in callback
    bridge = """package proof;
public class Bridge {
    public boolean run(net.minecraft.server.network.ServerPlayerEntity p) {
        return apply(p);
    }
""" + callback + "\n}\n"

    player = """package net.minecraft.server.network;
public class ServerPlayerEntity {
    private final java.util.UUID uuid;
    public ServerPlayerEntity(String id) {
        this.uuid = java.util.UUID.fromString(id);
    }
    public java.util.UUID getUuid() { return uuid; }
}
"""
    probe = """package proof;
public class Probe {
    public static void main(String[] args) {
        Bridge bridge = new Bridge();
        var a = new net.minecraft.server.network.ServerPlayerEntity(
          "00000000-0000-0000-0000-000000000001");
        var b = new net.minecraft.server.network.ServerPlayerEntity(
          "00000000-0000-0000-0000-000000000002");
        for (int i = 0; i < 4; i++) {
            if (!bridge.run(a)) throw new AssertionError("receipt");
        }
        if (!bridge.run(b)) throw new AssertionError("receipt B");
        long aBalance = AuthoredStateModel.readState_credits(
            java.util.Map.of("player", a.getUuid()));
        long bBalance = AuthoredStateModel.readState_credits(
            java.util.Map.of("player", b.getUuid()));
        if (aBalance != 20L || bBalance != 5L)
            throw new AssertionError("scoped balances: " + aBalance + "/" + bBalance);
        System.out.println("PLAYER_STATE_ATOMIC_PASS");
    }
}
"""
    sources = {
        "proof/AuthoredStateModel.java": state,
        "proof/Bridge.java": bridge,
        "proof/Probe.java": probe,
        "net/minecraft/server/network/ServerPlayerEntity.java": player,
    }
    paths = []
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        paths.append(str(path))
    compiled = subprocess.run(
        [javac, "-proc:none", "-d", str(tmp_path), *paths],
        capture_output=True, text=True, timeout=60, check=False)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(
        [java, "-cp", str(tmp_path), "proof.Probe"],
        capture_output=True, text=True, timeout=30, check=False)
    assert executed.returncode == 0, executed.stderr
    assert "PLAYER_STATE_ATOMIC_PASS" in executed.stdout
