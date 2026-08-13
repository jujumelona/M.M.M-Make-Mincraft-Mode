from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.external_agent_bridge import ExternalAgentBridge
from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall
from minecraft_mod_ai.model_adapters.llama_cpp_adapter import _parse_tool_calls
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

    def generate(self, request):  # pragma: no cover - tool path must use generate_turn
        raise AssertionError("text-only generate() must not run when tools are available")


def test_coder_calls_generation_tool_and_reinjects_result(monkeypatch) -> None:
    adapter = _Adapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )

    result = router.generate_text(
        "coder",
        [{"role": "user", "content": "Implement block registration"}],
    )

    assert result == "finished from tool evidence"
    assert runtime.schema_stages == ["generation"]
    assert runtime.calls == [
        ("generation", "search_code_rag", {"query": "register block"})
    ]
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
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )

    assert router.generate_text("coder", [{"role": "user", "content": "fix it"}]) == (
        "finished from tool evidence"
    )
    failure = json.loads(adapter.requests[1].messages[-1]["content"])
    assert failure["ok"] is False
    assert "provider unavailable" in failure["error"]


def test_external_bridge_tools_are_stage_scoped() -> None:
    generation = {
        item["function"]["name"]
        for item in ExternalAgentBridge.tool_schemas("generation")
    }
    assert generation == {
        "external_mcp_capabilities",
        "external_mcp_schema",
        "external_mcp_call",
    }
    assert ExternalAgentBridge.tool_schemas("release") == ()


def test_llama_openai_tool_call_parser_accepts_json_arguments() -> None:
    calls = _parse_tool_calls(
        [
            {
                "id": "call_map",
                "type": "function",
                "function": {
                    "name": "external_mcp_call",
                    "arguments": (
                        '{"capability":"mapping_resolution",'
                        '"arguments":{"class_name":"Block"}}'
                    ),
                },
            }
        ]
    )
    assert len(calls) == 1
    assert calls[0].id == "call_map"
    assert calls[0].name == "external_mcp_call"
    assert calls[0].arguments["capability"] == "mapping_resolution"


def test_agent_can_exceed_eight_tool_rounds(monkeypatch) -> None:
    class LongAdapter:
        def __init__(self) -> None:
            self.count = 0

        def generate_turn(self, request):
            self.count += 1
            if self.count <= 12:
                query = f"evidence_{self.count}"
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id=f"call_{self.count}",
                            name="search_code_rag",
                            arguments={"query": query},
                            raw_arguments=json.dumps({"query": query}),
                        ),
                    )
                )
            return GenerationResponse(content="enough evidence")

    adapter = LongAdapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    assert router.generate_text(
        "coder", [{"role": "user", "content": "research deeply"}]
    ) == "enough evidence"
    assert len(runtime.calls) == 12


def test_agent_synthesizes_final_answer_on_consecutive_exact_tool_fixed_point(
    monkeypatch,
) -> None:
    class LoopAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            if len(self.requests) <= 2:
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id=f"call_{len(self.requests)}",
                            name="search_code_rag",
                            arguments={"query": "same"},
                            raw_arguments='{"query":"same"}',
                        ),
                    )
                )
            assert request.tools == ()
            assert request.tool_choice is None
            assert request.parallel_tool_calls is False
            assert request.media_paths == ()
            assert request.messages[-1]["role"] == "system"
            assert "Tool use has converged" in request.messages[-1]["content"]
            return GenerationResponse(content="final answer from converged evidence")

    adapter = LoopAdapter()
    runtime = _ToolRuntime()
    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )
    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    assert router.generate_text(
        "coder", [{"role": "user", "content": "research"}]
    ) == "final answer from converged evidence"
    assert len(runtime.calls) == 2
    assert len(adapter.requests) == 3
