"""The real Fabric compiler probe follows the current host-owned callback ABI.

This test is offline. The expensive integration workflow separately downloads
verified Mojang/Fabric jars and invokes javac with the exact callback interfaces.
"""
from tools.verify_fabric_event_catalog import _CompilerProbeBodies


def test_compiler_probe_never_asks_small_model_for_raw_java_statement():
    provider = _CompilerProbeBodies()
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_predicate_try_0",
        parameters={"properties": {"expression": {"type": "string"}}},
    ) == {"expression": "true"}
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_operation_0_try_0",
        parameters={"properties": {"action": {"enum": ["call", "assign"]}}},
    ) == {"action": "call"}
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_operation_0_call_try_0",
        parameters={"properties": {"receiver": {"type": "string"}}},
    ) == {"receiver": "java.lang.System", "method": "nanoTime", "arguments": []}
    assert provider.generate_structured_data(
        "coder", [], data_name="submit_java_implementation_0_operation_1_try_0",
        parameters={"properties": {"action": {"enum": ["complete"]}}},
    ) == {"action": "complete"}


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
