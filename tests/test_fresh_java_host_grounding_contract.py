from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai import progress_aware_tool_loop as loop
from minecraft_mod_ai.grounding_policy import host_baseline_evidence_ready
from minecraft_mod_ai.host_grounding import _SCHEMA_VERSION as HOST_GROUNDING_SCHEMA
from minecraft_mod_ai.model_adapters import GenerationRequest

TARGET = "src/main/java/dev/mmm/debugfixture/DebugToken.java"


def _host_grounded_fresh_java_message() -> dict[str, object]:
    return {
        "role": "developer",
        "content": {
            "schema_version": "mmm/small-model-task-capsule",
            "reuse_action": "fresh",
            "mutation_target": {"path": TARGET},
            "primary_path": TARGET,
            "writable_paths": [TARGET],
            "host_grounding": {
                "schema_version": HOST_GROUNDING_SCHEMA,
                "policy": {
                    "resolved_before_first_coder_decode": True,
                    "baseline_grounding_owned_by_host": True,
                    "baseline_grounding_optional_for_model": False,
                    "model_tool_choice_required_for_baseline": False,
                    "writes_still_require_approved_pipeline": True,
                },
                "evidence_bindings": {
                    "project_exact_rag": {
                        "receipt": {
                            "project_sha256": "sha256:project",
                            "observations_sha256": "sha256:observations",
                        }
                    },
                    "approved_research_rag": {
                        "receipt": {
                            "selected_fact_count": 0,
                            "selected_record_count": 0,
                        }
                    },
                },
            },
        },
    }


def _tool(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_fresh_java_zero_research_facts_does_not_invalidate_host_grounding() -> None:
    message = _host_grounded_fresh_java_message()
    assert host_baseline_evidence_ready((message,)) is True


def test_unresolved_host_grounding_remains_not_ready() -> None:
    message = _host_grounded_fresh_java_message()
    content = message["content"]
    assert isinstance(content, dict)
    grounding = content["host_grounding"]
    assert isinstance(grounding, dict)
    policy = grounding["policy"]
    assert isinstance(policy, dict)
    policy["resolved_before_first_coder_decode"] = False

    assert host_baseline_evidence_ready((message,)) is False


class _StopFirstTurn(RuntimeError):
    pass


class _CapturingAdapter:
    def __init__(self) -> None:
        self.requests: list[GenerationRequest] = []

    def generate_turn(self, request: GenerationRequest):
        self.requests.append(request)
        raise _StopFirstTurn


def _fresh_owned_anchor_message_without_host_grounding() -> dict[str, object]:
    task_id = "task_debug_token"
    return {
        "role": "user",
        "content": {
            "phase": "implement_module",
            "task": "Implement the approved Minecraft/Fabric mod feature in the current project.",
            "module": {
                "module_id": task_id,
                "kind": "custom_java",
                "config": {
                    "evidence_task": {
                        "task_id": task_id,
                        "owned_anchors": [
                            {
                                "kind": "symbol",
                                "locator": f"{TARGET}#DebugToken",
                                "ownership": "exclusive",
                                "status": "host_reserved",
                                "module_id": ":",
                                "source_set": "main",
                            }
                        ],
                        "production_bindings": [
                            {
                                "task_ref": task_id,
                                "reuse_action": "fresh",
                                "owned_anchors": [
                                    {
                                        "kind": "symbol",
                                        "locator": f"{TARGET}#DebugToken",
                                        "ownership": "exclusive",
                                        "status": "host_reserved",
                                        "module_id": ":",
                                        "source_set": "main",
                                    }
                                ],
                            }
                        ],
                    }
                },
            },
        },
    }


def test_trusted_developer_fresh_target_still_obeys_explicit_evidence_policy(monkeypatch) -> None:
    adapter = _CapturingAdapter()
    monkeypatch.setattr(loop, "implementation_requested", lambda _messages: True)
    monkeypatch.setattr(loop, "mutation_history_applied", lambda _messages: False)
    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )

    request = GenerationRequest(
        messages=({**_fresh_owned_anchor_message_without_host_grounding(), "role": "developer"},),
        tools=(
            _tool("search_project_rag"),
            _tool("search_code_rag"),
            _tool("apply_source_edit"),
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    router = SimpleNamespace(_agent_require_fresh_evidence=True)
    runtime = SimpleNamespace()

    with pytest.raises(_StopFirstTurn):
        loop.generate_with_tools(
            router,
            config=SimpleNamespace(),
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )

    assert len(adapter.requests) == 1
    first = adapter.requests[0]
    assert [tool["function"]["name"] for tool in first.tools] == ["search_code_rag"]
    assert first.tool_choice == {
        "type": "function",
        "function": {"name": "search_code_rag"},
    }
    assert first.parallel_tool_calls is False


def test_untrusted_user_owned_anchor_cannot_bypass_generic_rag(monkeypatch) -> None:
    adapter = _CapturingAdapter()
    monkeypatch.setattr(loop, "implementation_requested", lambda _messages: True)
    monkeypatch.setattr(loop, "mutation_history_applied", lambda _messages: False)
    monkeypatch.setattr(
        loop,
        "fit_messages_to_context",
        lambda messages, *, config, tools: tuple(messages),
    )

    request = GenerationRequest(
        messages=(_fresh_owned_anchor_message_without_host_grounding(),),
        tools=(
            _tool("search_project_rag"),
            _tool("search_code_rag"),
            _tool("java_workspace_symbols"),
            _tool("external_mcp_call"),
            _tool("apply_source_edit"),
        ),
        tool_choice="auto",
        parallel_tool_calls=True,
    )
    router = SimpleNamespace(_agent_require_fresh_evidence=False)
    runtime = SimpleNamespace()

    with pytest.raises(_StopFirstTurn):
        loop.generate_with_tools(
            router,
            config=SimpleNamespace(),
            adapter=adapter,
            request=request,
            runtime=runtime,
            stage="generation",
            role="coder",
        )

    assert len(adapter.requests) == 1
    first = adapter.requests[0]
    assert [tool["function"]["name"] for tool in first.tools] == [
        "search_code_rag",
    ]
    assert first.tool_choice == {
        "type": "function",
        "function": {"name": "search_code_rag"},
    }
    assert first.parallel_tool_calls is False


@pytest.mark.parametrize("mode", ["first_turn", "provider_fallback", "exhausted"])
def test_materialized_fresh_authored_slot_uses_compile_probe_after_evidence_exhaustion(tmp_path, mode):
    from minecraft_mod_ai.model_adapters import (
        GenerationResponse,
        ModelConfigurationError,
        ToolCall,
    )
    from minecraft_mod_ai.mutation_authority import (
        CURRENT_MUTATION_AUTHORITY,
        MutationAuthority,
    )
    from minecraft_mod_ai.small_model_task_capsule_contract import (
        _CURRENT_CAPSULE,
        compile_task_capsule,
    )

    path = "src/main/java/demo/AuthoredFeature001.java"
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_text("package demo; public final class AuthoredFeature001 {\n"
                      " public static void initialize() { // MMM_AUTHORED_FEATURE_BODY_001\n }\n}",
                      encoding="utf-8")
    anchor = {"kind": "symbol", "locator": path + "#AuthoredFeature001",
              "status": "existing", "ownership": "host_exact_authored_lowering"}
    module = SimpleNamespace(module_id="authored_feature_001", kind="custom_java", config={
        "evidence_task": {"task_id": "authored_feature_001", "owned_anchors": [anchor],
                          "production_bindings": [{"task_ref": "authored_feature_001",
                                                   "reuse_action": "fresh", "owned_anchors": [anchor]}],
                          "required_gates": ["target_compile"]}})
    capsule = compile_task_capsule(module)
    operations = []

    class EvidenceAdapter(_CapturingAdapter):
        def generate_turn(self, request):
            self.requests.append(request)
            assert len(self.requests) <= 8
            schema = request.tools[0]["function"]
            if mode == "first_turn" or schema["name"] == "apply_source_edit":
                raise _StopFirstTurn
            name = schema["name"]
            args = {}
            if name in {"external_mcp_schema", "external_mcp_call"}:
                args["capability"] = schema["parameters"]["properties"]["capability"]["enum"][0]
            return GenerationResponse(tool_calls=(ToolCall(id=str(len(self.requests)), name=name, arguments=args),))

    def call(stage, name, args, **kwargs):
        assert stage == "generation"
        operations.append((name, args.get("capability")))
        if name == "search_code_rag":
            return {"schema_version": "mmm/code-rag-result-v1", "hits": []}
        if name == "external_mcp_capabilities":
            return {"capabilities": {"source_search": [{}], "official_mod_docs": [{}]}}
        if name == "external_mcp_schema":
            return {"status": "PASS"}
        assert name == "external_mcp_call"
        cap = args["capability"]
        if cap == "source_search" or mode == "exhausted":
            return {"schema_version": "mmm/external-mcp-evidence-bundle-v1",
                    "capability": cap, "status": "UNAVAILABLE", "evidence": []}
        return {"schema_version": "mmm/external-mcp-evidence-bundle-v1", "capability": cap,
                "status": "PASS", "evidence": [{
                    "schema_version": "mmm/external-mcp-call-receipt-v1", "capability": cap,
                    "status": "PASS", "access": "read", "result": {"text": [
                        "import net.minecraft.world.level.block.Block;\npublic class LaunchPad extends Block {}"
                    ]}}]}

    adapter = EvidenceAdapter()
    tools = [_tool("search_code_rag"), _tool("apply_source_edit")]
    if mode != "first_turn":
        for name in ("external_mcp_capabilities", "external_mcp_schema", "external_mcp_call"):
            tool = _tool(name)
            tool["function"]["parameters"]["properties"] = {
                "capability": {"type": "string"}, "arguments": {"type": "object"}}
            tools.append(tool)
    token = _CURRENT_CAPSULE.set(capsule)
    authority = CURRENT_MUTATION_AUTHORITY.set(MutationAuthority.exact((path,)))
    try:
        expected = pytest.raises(_StopFirstTurn)
        with expected:
            loop.generate_with_tools(
                SimpleNamespace(_agent_require_fresh_evidence=True),
                config=SimpleNamespace(max_context=32768, max_new_tokens=4096), adapter=adapter,
                request=GenerationRequest(messages=(
                    {"role": "developer", "content": capsule.to_host_authority_payload()},
                    {"role": "user", "content": {"phase": "implement_module", "task": "Register the launch pad using the selected target API."}},
                ), tools=tuple(tools)),
                runtime=SimpleNamespace(workspace_root=tmp_path, call=call, call_scoped=call), stage="generation", role="coder",
            )
    finally:
        _CURRENT_CAPSULE.reset(token)
        CURRENT_MUTATION_AUTHORITY.reset(authority)
    assert [tool["function"]["name"] for tool in adapter.requests[0].tools] == ["search_code_rag"]
    if mode != "first_turn":
        assert operations == [
            ("search_code_rag", None), ("external_mcp_capabilities", None),
            ("external_mcp_schema", "source_search"), ("external_mcp_call", "source_search"),
            ("external_mcp_schema", "official_mod_docs"), ("external_mcp_call", "official_mod_docs"),
        ]
    if mode in {"provider_fallback", "exhausted"}:
        # Once all reviewed evidence routes are exhausted, the host exposes only
        # the exact task-owned mutation action. Completion still requires the
        # downstream target_compile gate; this is a bounded compile probe, not
        # evidence-free acceptance.
        assert adapter.requests[-1].tools[0]["function"]["name"] == "apply_source_edit"
