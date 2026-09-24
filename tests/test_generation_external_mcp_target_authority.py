from __future__ import annotations

from jsonschema import Draft202012Validator

from minecraft_mod_ai import generation_evidence_controller as evidence_controller
from minecraft_mod_ai import progress_aware_tool_loop as tool_loop
from minecraft_mod_ai.agent_tool_runtime import AgentToolRuntime
from minecraft_mod_ai.external_agent_bridge import ExternalAgentBridge
from minecraft_mod_ai.external_mcp_recovery_contract import (
    constrain_recovery_tools,
    record_discovery,
)
from minecraft_mod_ai.model_adapters import ToolCall


def _schema(name: str) -> dict:
    return next(
        item["function"]["parameters"]
        for item in ExternalAgentBridge.tool_schemas("generation")
        if item["function"]["name"] == name
    )


def test_generation_capability_schema_accepts_model_version_scalar_before_host_binding() -> None:
    schema = _schema("external_mcp_capabilities")

    # Regression for the Debug Mode trace where the small model emitted 26.2 as
    # a JSON number. Generation target coordinates are host-owned, so their
    # model representation must not reject the call before host normalization.
    Draft202012Validator(schema).validate(
        {
            "minecraft_version": 26.2,
            "loader": "fabric",
            "mappings": "official",
            "max_access": "read",
        }
    )


def test_generation_external_mcp_call_uses_project_platform_lock(
    tmp_path,
    monkeypatch,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "build.gradle").write_text("// fixture\n", encoding="utf-8")
    metadata = tmp_path / ".minecraft_ai"
    metadata.mkdir()
    (metadata / "platform-lock.json").write_text(
        """{
  "minecraft_version": "26.2",
  "loader": "fabric",
  "mappings_kind": "official",
  "mappings_version": "",
  "yarn_mappings": ""
}
""",
        encoding="utf-8",
    )

    runtime = AgentToolRuntime(profile="test", workspace_root=tmp_path)
    captured: dict[str, object] = {}

    def fake_call(stage, name, payload, *, allowed_server_ids=None):
        captured["stage"] = stage
        captured["name"] = name
        captured["payload"] = dict(payload)
        captured["allowed_server_ids"] = allowed_server_ids
        return {"status": "PASS"}

    monkeypatch.setattr(runtime._external_bridge, "call", fake_call)

    runtime.call(
        "generation",
        "external_mcp_capabilities",
        {
            "minecraft_version": 1.20,
            "loader": "forge",
            "mappings": "yarn",
            "max_access": "admin",
        },
    )

    assert captured["stage"] == "generation"
    assert captured["name"] == "external_mcp_capabilities"
    assert captured["payload"] == {
        "minecraft_version": "26.2",
        "loader": "fabric",
        "mappings": "official",
        "max_access": "read",
    }



def _internal_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    }


def test_reviewed_alternate_retriever_is_host_normalized_without_consuming_route() -> None:
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.OBSERVE)
    rejected = ToolCall(
        id="rejected-1",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "search_project_rag",
            "raw_arguments": '{"query":"block registration 26.2","limit":8}',
        },
        raw_arguments="{}",
    )

    normalized = evidence_controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(_internal_schema("search_code_rag"),),
        forced_evidence_tool="search_code_rag",
    )

    assert normalized is not None
    assert normalized[0].name == "search_code_rag"
    assert normalized[0].arguments == {
        "query": "block registration 26.2",
        "limit": 8,
    }
    assert state.attempted_sources == set()


def test_nonreviewed_invisible_tool_cannot_consume_forced_evidence_route() -> None:
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.OBSERVE)
    rejected = ToolCall(
        id="rejected-2",
        name="__mmm_rejected_tool_call__",
        arguments={
            "failure_code": "TOOL_NOT_VISIBLE",
            "original_tool": "java_file_read",
            "raw_arguments": '{"query":"Foo"}',
        },
        raw_arguments="{}",
    )

    normalized = evidence_controller.normalize_forced_evidence_rejection_calls(
        (rejected,),
        phase_tools=(_internal_schema("search_project_rag"),),
        forced_evidence_tool="search_project_rag",
    )

    assert normalized is None
    assert state.attempted_sources == set()
    assert not hasattr(tool_loop, "_consume_rejected_evidence_fixed_point")

def _external_schema(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["capability"],
            },
        },
    }


def test_recovery_mcp_schema_is_narrowed_to_reviewed_discovered_capability() -> None:
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.RECOVER)
    call = ToolCall(
        id="caps",
        name="external_mcp_capabilities",
        arguments={},
        raw_arguments="{}",
    )
    record_discovery(
        state,
        call,
        {
            "ok": True,
            "result": {
                "capabilities": {
                    "source_search": [{}],
                    "read_file": [{}],
                    "official_mod_docs": [{}],
                }
            },
        },
        external_rag_capability=lambda value: (
            str(value.get("capability"))
            if value.get("capability") in {"source_search", "official_mod_docs"}
            else ""
        ),
    )

    narrowed = constrain_recovery_tools(
        (_external_schema("external_mcp_schema"),),
        state=state,
        repair_route="official_api",
    )

    capability = narrowed[0]["function"]["parameters"]["properties"]["capability"]
    assert capability["enum"] == ["source_search"]


def test_recovery_mcp_call_requires_successful_schema_binding() -> None:
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.RECOVER)
    state._external_mcp_capabilities_seen = True
    state._external_mcp_recovery_capabilities = "source_search",

    assert constrain_recovery_tools(
        (_external_schema("external_mcp_call"),),
        state=state,
        repair_route="official_api",
    ) == ()

    schema_call = ToolCall(
        id="schema",
        name="external_mcp_schema",
        arguments={"capability": "source_search"},
        raw_arguments='{"capability":"source_search"}',
    )
    record_discovery(
        state,
        schema_call,
        {"ok": True, "result": {"status": "PASS"}},
        external_rag_capability=lambda value: str(value.get("capability") or ""),
    )
    narrowed = constrain_recovery_tools(
        (_external_schema("external_mcp_call"),),
        state=state,
        repair_route="official_api",
    )
    assert narrowed[0]["function"]["parameters"]["properties"]["capability"]["enum"] == [
        "source_search"
    ]


def test_fabric_api_diagnostic_prefers_loader_docs_over_vanilla_source():
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.RECOVER,
        latest_verifier_errors=({"message": "package net.fabricmc.fabric.api.client.itemgroup.v1 does not exist"},))
    state._external_mcp_capabilities_seen = True
    state._external_mcp_recovery_capabilities = ("source_search", "official_mod_docs")
    narrowed = constrain_recovery_tools((_external_schema("external_mcp_schema"),),
        state=state, repair_route="official_api")
    assert narrowed[0]["function"]["parameters"]["properties"]["capability"]["enum"] == ["official_mod_docs"]


def test_failed_provider_does_not_exhaust_other_discovered_capabilities():
    tools = tuple(_external_schema(name) for name in (
        "external_mcp_capabilities", "external_mcp_schema", "external_mcp_call"))
    state = tool_loop.HostRunState(phase=tool_loop.LoopPhase.RECOVER)
    state.record_source_attempt("external_mcp_capabilities", {})
    record_discovery(state, ToolCall(id="caps", name="external_mcp_capabilities", arguments={}),
                     {"ok": True, "result": {"capabilities": {
                         "source_search": [{}], "official_mod_docs": [{}]}}},
                     external_rag_capability=lambda value: value["capability"])
    called = []
    for capability in ("source_search", "official_mod_docs"):
        for tool_name in ("external_mcp_schema", "external_mcp_call"):
            frontier = tool_loop._filter_tools_for_phase(
                tools, state.phase, "coder", attempted_sources=state.attempted_sources,
                repair_evidence_route="official_api")
            selected = constrain_recovery_tools(frontier, state=state, repair_route="official_api",
                                                available_tools=tools)
            assert len(selected) == 1
            assert selected[0]["function"]["name"] == tool_name
            assert selected[0]["function"]["parameters"]["properties"]["capability"]["enum"] == [capability]
            args = {"capability": capability}
            call = ToolCall(id=str(len(called)), name=tool_name, arguments=args)
            state.record_source_attempt(tool_name, args)
            record_discovery(state, call, {"ok": True, "result": {
                "status": "PASS" if tool_name == "external_mcp_schema" else "UNAVAILABLE"}},
                external_rag_capability=lambda value: value["capability"])
            called.append((tool_name, capability))
    assert len(called) == 4
    assert constrain_recovery_tools((), state=state, repair_route="official_api",
                                    available_tools=tools) == ()
    state.phase = tool_loop.LoopPhase.ACT
    assert constrain_recovery_tools((), state=state, repair_route="official_api",
                                    available_tools=tools) == ()
