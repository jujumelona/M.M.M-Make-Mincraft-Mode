"""The real Fabric compiler probe follows the current host-owned callback ABI.

This test is offline. The expensive integration workflow separately downloads
verified Mojang/Fabric jars and invokes javac with the exact callback interfaces.
"""
import pytest

from tools.verify_fabric_event_catalog import _CompilerProbeBodies


def test_probe_uses_owned_state_when_callback_has_no_mutator():
    from minecraft_mod_ai.first_pass_java_source import compose_first_pass_java
    from test_host_fabric_event_catalog import event_symbols

    symbols = event_symbols()
    symbols["net.minecraft.server.MinecraftServer"] = {
        "kind": "CLASS", "owner": "net.minecraft.server.MinecraftServer", "access": 1}
    source = compose_first_pass_java(_CompilerProbeBodies(), {
        "language": "java", "side": "COMMON", "target_path": "src/main/java/proof/Probe.java",
        "bindings": {"request_feature_entrypoint": "proof.Probe"},
    }, [], construction_context={"symbols": symbols, "host_state_access": [{
        "owner": "proof.CompilerProbeState", "methods": [{
            "name": "writeState_abi_probe_count",
            "signature": "public static void writeState_abi_probe_count(int value);",
        }],
    }]}, behaviors=[{
        "behavior_id": "compiler-probe", "intent": "Compiler ABI probe",
        "trigger": "The selected callback fires", "effects": "Compiler probe only",
        "event_binding": "net.fabricmc.fabric.api.event.lifecycle.v1.ServerTickEvents#END_SERVER_TICK",
        "state_writes": ["abi_probe_count"],
    }])
    assert "proof.CompilerProbeState.writeState_abi_probe_count(1);" in source


def test_compiler_probe_never_asks_small_model_for_raw_java_statement():
    provider = _CompilerProbeBodies()
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_predicate_try_0",
        parameters={"properties": {"expression": {"type": "string"}}},
    ) == {"expression": "true"}
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_operation_0_try_0",
        parameters={"properties": {"action": {"enum": ["assign", "complete"]}}},
    ) == {"action": "complete"}
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_required_state_abi_probe_count_try_0",
        parameters={"properties": {"expression": {"type": "string"}}},
    ) == {"expression": "1"}
    with pytest.raises(AssertionError, match="declared state writer"):
        provider.generate_structured_data(
            "coder", [], data_name="submit_java_implementation_0_operation_0_try_0",
            parameters={"properties": {"action": {"enum": ["call", "assign"]}}},
        )


def test_compiler_probe_uses_inspected_result_type_and_actual_enum_choices():
    provider = _CompilerProbeBodies()
    import json
    event = {"verified_callback": {"return_type": "example.Result"}}
    messages = [{"role": "system", "content": "Select result"}, {
        "role": "user", "content": json.dumps(event),
    }]
    schema = {"properties": {"expression": {"enum": [
        "example.Result.PASS", "example.Result.SUCCESS",
    ]}}}
    assert provider.generate_structured_data(
        "coder", messages, data_name="select_callback_result_unhandled",
        parameters=schema,
    ) == {"expression": "example.Result.PASS"}
    assert provider.generate_structured_data(
        "coder", messages, data_name="select_callback_result_handled",
        parameters=schema,
    ) == {"expression": "example.Result.SUCCESS"}
    schema = {"properties": {"expression": {"type": "string"}}}
    event["verified_callback"]["return_type"] = "boolean"
    messages[1]["content"] = json.dumps(event)
    assert provider.generate_structured_data(
        "coder", messages, data_name="select_callback_result_handled",
        parameters=schema,
    ) == {"expression": "true"}
