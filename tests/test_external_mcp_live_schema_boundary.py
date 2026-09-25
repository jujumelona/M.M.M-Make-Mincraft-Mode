from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from minecraft_mod_ai import external_agent_bridge, external_mcp_router
from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime
from minecraft_mod_ai.external_mcp import ExternalMCPRegistry
from minecraft_mod_ai.external_mcp_binding_contract import (
    ExternalMCPArgumentsError,
    install,
)
from minecraft_mod_ai.external_mcp_recovery_contract import (
    constrain_recovery_tools,
    record_discovery,
)
from minecraft_mod_ai.model_adapters import ToolCall
from minecraft_mod_ai.progress_aware_tool_loop import HostRunState


def test_real_stdio_schema_rejects_bad_arguments_then_accepts_host_bound_target(tmp_path, monkeypatch):
    monkeypatch.delenv("MMM_ROOT_CAUSE_TRACE_DETAIL", raising=False)
    trace_path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("MMM_ROOT_CAUSE_TRACE_PATH", str(trace_path))
    invocations = tmp_path / "calls.jsonl"
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump({
        "schema_version": "mmm/external-mcp-registry-v2",
        "servers": {"probe": {
            "status": "enabled", "transport": "stdio",
            "command": [sys.executable, str(Path(__file__).parent / "fixtures" / "mcp_schema_probe.py"),
                        str(invocations)],
            "version_policy": "dynamic", "loaders": ["fabric"], "trust": "test",
            "capabilities": {"source_search": {
                "tool": "search", "access": "read", "stages": ["generation"],
                "priority": 10, "target_args": {"minecraft_version": "version"},
            }},
        }},
    }), encoding="utf-8")
    install(external_agent_bridge, external_mcp_router)
    bridge = external_agent_bridge.ExternalAgentBridge()
    bridge._router = external_mcp_router.ExternalMCPRouter(ExternalMCPRegistry(registry_path))
    payload = {"capability": "source_search", "minecraft_version": "26.2",
               "loader": "fabric", "mappings": "official", "max_access": "read"}
    schema_result = bridge.call("generation", "external_mcp_schema", payload)
    assert schema_result["status"] == "PASS", json.dumps(schema_result, indent=2)
    state = HostRunState()
    state._external_mcp_capabilities_seen = True
    state._external_mcp_recovery_capabilities = ("source_search",)
    record_discovery(state, ToolCall(id="schema", name="external_mcp_schema", arguments=payload),
                     {"ok": True, "result": schema_result}, external_rag_capability=lambda args: args["capability"])
    tool = next(tool for tool in bridge.tool_schemas("generation")
                if tool["function"]["name"] == "external_mcp_call")
    projected = constrain_recovery_tools([tool], state=state, repair_route=None)[0]
    validator = Draft202012Validator(projected["function"]["parameters"])
    validator.validate({"capability": "source_search", "arguments": {"query": "Item registry"}})
    assert list(validator.iter_errors({"capability": "source_search",
                                       "arguments": {"path": "AuthoredFeature001.java", "operation": "read"}}))
    with pytest.raises(ExternalMCPArgumentsError, match="EXTERNAL_MCP_ARGUMENTS_INVALID"):
        bridge.call("generation", "external_mcp_call", {
            **payload, "arguments": {"path": "AuthoredFeature001.java", "operation": "read"},
        })
    assert not invocations.exists()
    result = bridge.call("generation", "external_mcp_call", {
        **payload, "arguments": {"query": "Item registry", "version": "wrong model version"},
    })
    assert result["status"] == "PASS"
    assert json.loads(invocations.read_text(encoding="utf-8")) == {
        "query": "Item registry", "version": "26.2",
    }
    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    runtime._external_bridge = bridge
    failed = runtime.call("generation", "external_mcp_call", {
        **payload, "arguments": {"query": "provider failure"},
    })
    assert failed["status"] == "UNAVAILABLE"
    assert failed["evidence"] == []
    assert "deliberate provider" in json.dumps(failed["attempts"])
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    validations = [event for event in events if event["event"] == "external_mcp_arguments_validated"]
    assert [event["result"] for event in validations] == ["FAIL", "PASS", "PASS"]
    assert validations[0]["details"]["errors"]
    assert validations[1]["details"]["arguments"] == {"query": "Item registry", "version": "26.2"}
    runtime_result = next(event for event in events if event["event"] == "agent_tool_call_result")
    assert runtime_result["result"] == "FAIL"
    assert runtime_result["details"]["result"]["status"] == "UNAVAILABLE"
    assert any(event["event"] == "mcp_child_stderr" for event in events)


@pytest.mark.parametrize("schema_id", [None, "https://example.test/probe-schema"])
def test_projected_provider_refs_and_argument_retry_keep_the_live_contract(schema_id):
    from minecraft_mod_ai.generation_loop_outcomes import runtime_failure_code

    state = HostRunState()
    state._external_mcp_capabilities_seen = True
    state._external_mcp_recovery_capabilities = ("source_search",)
    schema = {"type": "object", "$defs": {"query": {"type": "string", "minLength": 1}},
              "properties": {"query": {"$ref": "#/$defs/query"}}, "required": ["query"]}
    if schema_id:
        schema["$id"] = schema_id
    record_discovery(state, ToolCall(id="schema", name="external_mcp_schema",
                                     arguments={"capability": "source_search"}),
                     {"ok": True, "result": {"status": "PASS", "input_schema": schema}},
                     external_rag_capability=lambda args: args["capability"])
    failure_code = runtime_failure_code("external_mcp_call", "ExternalMCPArgumentsError: EXTERNAL_MCP_ARGUMENTS_INVALID")
    record_discovery(state, ToolCall(id="bad", name="external_mcp_call",
                                     arguments={"capability": "source_search", "arguments": {"path": "x"}}),
                     {"ok": False, "failure_code": failure_code},
                     external_rag_capability=lambda args: args["capability"])
    tool = next(tool for tool in external_agent_bridge.ExternalAgentBridge.tool_schemas("generation")
                if tool["function"]["name"] == "external_mcp_call")
    projected = constrain_recovery_tools([tool], state=state, repair_route=None)[0]
    validator = Draft202012Validator(projected["function"]["parameters"])
    validator.validate({"capability": "source_search", "arguments": {"query": "Item registry"}})
    assert list(validator.iter_errors({"capability": "source_search", "arguments": {"query": ""}}))
