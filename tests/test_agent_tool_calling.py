from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from minecraft_mod_ai.external_agent_bridge import ExternalAgentBridge
from minecraft_mod_ai.model_adapters import (
    GenerationResponse,
    ModelConfigurationError,
    ToolCall,
)
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _parse_qwen_tool_markup
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def __init__(self) -> None:
        self.config = SimpleNamespace(adapter="llama_cpp", exclusive_gpu=False)

    def load_profile(self, profile: str) -> None:
        assert profile == "test"

    def role(self, profile: str, role: str):
        assert profile == "test"
        assert role in {"coder", "planner"}
        return self.config


class _ToolRuntime:
    def __init__(self) -> None:
        self.schema_stages: list[str] = []
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def tool_schemas(self, stage: str):
        self.schema_stages.append(stage)
        return (
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "Search project code",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        )

    def call(self, stage: str, name: str, arguments):
        payload = dict(arguments)
        self.calls.append((stage, name, payload))
        return {"hits": [{"path": "Example.java", "line": 7}]}


class _Adapter:
    def __init__(self) -> None:
        self.requests = []

    def generate_turn(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="call_1",
                        name="search_code_rag",
                        arguments={"query": "register block"},
                        raw_arguments='{"query":"register block"}',
                    ),
                )
            )
        return GenerationResponse(content="finished from tool evidence")

    def generate(self, request):
        raise AssertionError("text-only generate() must not run when tools are available")


def test_coder_calls_generation_tool_and_reinjects_result(monkeypatch) -> None:
    adapter = _Adapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)
    result = router.generate_text("coder", [{"role": "user", "content": "Implement block registration"}])
    assert result == "finished from tool evidence"
    assert runtime.schema_stages == ["generation"]
    assert runtime.calls == [("generation", "search_code_rag", {"query": "register block"})]
    assert len(adapter.requests) == 2
    assert adapter.requests[0].tool_choice == "auto"
    assert adapter.requests[0].parallel_tool_calls is True
    assert adapter.requests[0].tools[0]["function"]["name"] == "search_code_rag"
    reinjected = adapter.requests[1].messages
    assistant = reinjected[-2]
    tool_result = reinjected[-1]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert tool_result["role"] == "tool"
    assert tool_result["tool_call_id"] == "call_1"
    decoded = json.loads(tool_result["content"])
    assert decoded["ok"] is True
    assert decoded["result"]["hits"][0]["path"] == "Example.java"


def test_tool_failure_is_observation_and_model_can_finish(monkeypatch) -> None:
    adapter = _Adapter()

    class BrokenRuntime(_ToolRuntime):
        def call(self, stage: str, name: str, arguments):
            raise RuntimeError("provider unavailable")

    runtime = BrokenRuntime()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)
    assert router.generate_text("coder", [{"role": "user", "content": "fix it"}]) == "finished from tool evidence"
    failure = json.loads(adapter.requests[1].messages[-1]["content"])
    assert failure["ok"] is False
    assert "provider unavailable" in failure["error"]


def test_external_bridge_tools_are_stage_scoped() -> None:
    generation = {item["function"]["name"] for item in ExternalAgentBridge.tool_schemas("generation")}
    assert generation == {"external_mcp_capabilities", "external_mcp_schema", "external_mcp_call"}
    assert ExternalAgentBridge.tool_schemas("release") == ()


def test_llama_qwen_tool_markup_parser_accepts_schema_typed_arguments() -> None:
    schemas = {
        "external_mcp_call": {
            "type": "object",
            "properties": {
                "capability": {"type": "string"},
                "arguments": {"type": "object"},
            },
            "required": ["capability", "arguments"],
            "additionalProperties": False,
        }
    }
    text = (
        "<tool_call><function=external_mcp_call>"
        "<parameter=capability>mapping_resolution</parameter>"
        '<parameter=arguments>{"class_name":"Block"}</parameter>'
        "</function></tool_call>"
    )
    visible, calls = _parse_qwen_tool_markup(text, schemas)
    assert visible == ""
    assert len(calls) == 1
    assert calls[0].name == "external_mcp_call"
    assert calls[0].arguments == {"capability": "mapping_resolution", "arguments": {"class_name": "Block"}}


def test_agent_can_exceed_eight_tool_rounds_when_evidence_keeps_changing(monkeypatch) -> None:
    class LongAdapter:
        def __init__(self) -> None:
            self.count = 0

        def generate_turn(self, request):
            self.count += 1
            if self.count <= 12:
                query = f"evidence_{self.count}"
                return GenerationResponse(
                    tool_calls=(ToolCall(id=f"call_{self.count}", name="search_code_rag", arguments={"query": query}, raw_arguments=json.dumps({"query": query})),)
                )
            return GenerationResponse(content="enough evidence")

    class NovelRuntime(_ToolRuntime):
        def call(self, stage: str, name: str, arguments):
            payload = dict(arguments)
            self.calls.append((stage, name, payload))
            return {"hits": [{"path": f"{payload['query']}.java", "line": len(self.calls)}]}

    adapter = LongAdapter()
    runtime = NovelRuntime()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)
    assert router.generate_text("coder", [{"role": "user", "content": "research deeply"}]) == "enough evidence"
    assert len(runtime.calls) == 12


def test_duplicate_retrieval_query_is_not_executed_twice(monkeypatch) -> None:
    class LoopAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            if len(self.requests) <= 2:
                return GenerationResponse(
                    tool_calls=(ToolCall(id=f"call_{len(self.requests)}", name="search_code_rag", arguments={"query": "same"}, raw_arguments='{"query":"same"}'),)
                )
            assert [item["function"]["name"] for item in request.tools] == ["search_code_rag"]
            assert request.tool_choice == "auto"
            assert request.parallel_tool_calls is True
            assert request.media_paths == ()
            return GenerationResponse(content="final answer from converged evidence")

    adapter = LoopAdapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)
    assert router.generate_text("coder", [{"role": "user", "content": "research"}]) == "final answer from converged evidence"
    assert len(runtime.calls) == 1
    assert len(adapter.requests) == 3


def test_host_owned_grounding_satisfies_baseline_without_forced_rag(monkeypatch) -> None:
    class FinalAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            return GenerationResponse(content="implemented from host grounding")

    grounding = {
        "schema_version": "mmm/host-owned-coder-grounding-v1",
        "evidence_bindings": {
            "project_exact_rag": {
                "receipt": {
                    "project_sha256": "sha256:project",
                    "observations_sha256": "sha256:observations",
                }
            }
        },
        "policy": {
            "resolved_before_first_coder_decode": True,
            "baseline_grounding_owned_by_host": True,
            "baseline_grounding_optional_for_model": False,
            "model_tool_choice_required_for_baseline": False,
            "supplemental_retrieval_after_host_validation": True,
        },
    }
    adapter = FinalAdapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)
    router._agent_require_fresh_evidence = True
    result = router.generate_text("coder", [{"role": "user", "content": json.dumps({"host_grounding": grounding})}])
    assert result == "implemented from host grounding"
    assert runtime.calls == []
    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert [item["function"]["name"] for item in request.tools] == ["search_code_rag"]
    assert request.tool_choice == "auto"
    assert request.parallel_tool_calls is True


def test_required_rag_exhaustion_fails_before_hard_round_budget(monkeypatch) -> None:
    class WeakRuntime(_ToolRuntime):
        def call(self, stage: str, name: str, arguments):
            payload = dict(arguments)
            self.calls.append((stage, name, payload))
            return {"receipt": {"result_count": 0, "coverage_score": 0.0, "relevance_score": 0.0}}

    class ExhaustAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return GenerationResponse(content="draft without evidence")
            if len(self.requests) == 2:
                assert request.tool_choice == {"type": "function", "function": {"name": "search_code_rag"}}
                return GenerationResponse(
                    tool_calls=(ToolCall(id="forced_1", name="search_code_rag", arguments={"query": "missing registration api"}, raw_arguments='{"query":"missing registration api"}'),)
                )
            return GenerationResponse(content="no reviewed evidence route remains")

    adapter = ExhaustAdapter()
    runtime = WeakRuntime()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry(), agent_tool_runtime_factory=lambda **_: runtime)
    router._agent_require_fresh_evidence = True
    with pytest.raises(ModelConfigurationError, match="Required production evidence is unavailable"):
        router.generate_text("coder", [{"role": "user", "content": "implement unknown API"}])
    assert len(runtime.calls) == 1
    assert len(adapter.requests) == 3


def test_main_only_policy_is_injected_even_when_tools_are_disabled(monkeypatch) -> None:
    class CapturingAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            return "done"

    adapter = CapturingAdapter()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry())

    assert router.generate_text(
        "coder",
        [{"role": "user", "content": "make the requested repository change"}],
        enable_tools=False,
    ) == "done"

    assert len(adapter.requests) == 1
    system_messages = [
        str(message.get("content", ""))
        for message in adapter.requests[0].messages
        if message.get("role") == "system"
    ]
    assert any("The only permitted Git branch/ref" in content for content in system_messages)
    assert any("Never call any branch-creation action" in content for content in system_messages)
    assert any("`main`" in content for content in system_messages)


def test_main_only_policy_is_injected_into_native_tool_decisions(monkeypatch) -> None:
    class DecisionAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="decision_1",
                        name="choose_action",
                        arguments={"choice": "safe"},
                        raw_arguments='{"choice":"safe"}',
                    ),
                )
            )

    adapter = DecisionAdapter()
    monkeypatch.setattr(ModelRouter, "_new_text_adapter", staticmethod(lambda config, *, role: adapter))
    router = ModelRouter(profile="test", registry=_Registry())

    assert router.generate_tool_decision(
        "coder",
        [{"role": "user", "content": "choose the next action"}],
        tool_name="choose_action",
        parameters={
            "type": "object",
            "properties": {"choice": {"type": "string"}},
            "required": ["choice"],
            "additionalProperties": False,
        },
    ) == {"choice": "safe"}

    assert len(adapter.requests) == 1
    system_messages = [
        str(message.get("content", ""))
        for message in adapter.requests[0].messages
        if message.get("role") == "system"
    ]
    assert any("The only permitted Git branch/ref" in content for content in system_messages)
    assert any("Never call any branch-creation action" in content for content in system_messages)
    assert any("`main`" in content for content in system_messages)
