"""State API diagnostics point at the offending operation, without catalog prose."""
from __future__ import annotations

import re

import pytest

from test_incremental_java_dependency_repair import _setup


@pytest.mark.parametrize("earlier", [
    {"id": "earlier", "op": "call", "receiver": "sample.State", "method": "readState_balance", "arguments": [], "result_type": ""},
    {"id": "earlier", "op": "call", "receiver": "System.out", "method": "println", "arguments": ['"getState"'], "result_type": ""},
    {"id": "earlier", "op": "call", "receiver": "sample.Other", "method": "getState", "arguments": [], "result_type": ""},
])
def test_raw_state_call_is_mapped_after_an_earlier_valid_call_or_literal(earlier):
    from minecraft_mod_ai.incremental_java_author import IncrementalJavaBuilder
    from minecraft_mod_ai.incremental_java_state import validate_program_state
    from minecraft_mod_ai.host_typed_state_access import validate_host_state_access, validate_behavior_state_provenance
    body = [earlier,
            {"id": "raw", "op": "call", "receiver": "sample.State", "method": "getState",
             "arguments": [], "result_type": ""}]
    _, program, spec, context, messages, behaviors = _setup(body)
    program["members"][0]["behavior"] = {"behavior_id": "claim", "state_reads": ["balance"]}
    context["host_state_access"] = [{"owner": "sample.State", "methods": [
        {"name": "readState_balance", "parameters": [], "return_type": "int"}]}]
    context["incremental_state"]["program"] = program
    builder = IncrementalJavaBuilder(None, spec, messages, context, behaviors, program)
    source, _ = builder.render()
    with pytest.raises(ValueError, match="JAVA_HOST_RAW_STATE_ACCESS") as failed:
        validate_program_state(source, spec, context,
                               validators=(validate_host_state_access, validate_behavior_state_provenance))
    diagnostic = str(failed.value)
    line = int(re.search(r"Demo.java:(\d+):", diagnostic)[1])
    assert builder.line_map[line]["node_id"] == "raw"
    assert builder.line_map[line]["slot"] == "method"
    assert "readState_balance" not in diagnostic
