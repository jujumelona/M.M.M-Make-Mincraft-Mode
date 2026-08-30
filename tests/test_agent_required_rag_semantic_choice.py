from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def __init__(self) -> None:
        self.config = SimpleNamespace(adapter="llama_cpp", exclusive_gpu=False)

    def load_profile(self, profile: str) -> None:
        assert profile == "test"

    def role(self, profile: str, role: str):
        assert profile == "test"
        assert role == "coder"
        return self.config


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def tool_schemas(self, stage: str):
        assert stage == "generation"
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": name,
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
            for name in ("search_code_rag", "search_project_rag")
        )

    def call(self, stage: str, name: str, arguments):
        payload = dict(arguments)
        self.calls.append((stage, name, payload))
        return {
            "receipt": {
                "result_count": 1,
                "coverage_score": 1.0,
                "relevance_score": 1.0,
            },
            "hits": [
                {
                    "path": "src/main/java/example/Example.java",
                    "text": "public final class Example { public static void register() {} }",
                }
            ],
        }


class _Adapter:
    def __init__(self) -> None:
        self.requests = []

    def generate_turn(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            assert request.tool_choice == "auto"
            return GenerationResponse(content="I should answer, but I have not gathered evidence yet.")
        if len(self.requests) == 2:
            names = {item["function"]["name"] for item in request.tools}
            assert names == {"search_code_rag", "search_project_rag"}
            assert request.tool_choice == "required"
            assert request.parallel_tool_calls is False
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="model_choice_1",
                        name="search_project_rag",
                        arguments={"query": "exact project registration contract"},
                        raw_arguments=json.dumps(
                            {"query": "exact project registration contract"}
                        ),
                    ),
                )
            )
        assert len(self.requests) == 3
        return GenerationResponse(content="finished from model-selected project evidence")

    def generate(self, request):
        raise AssertionError("native tool loop must remain active")


def test_required_evidence_does_not_force_one_semantic_rag_route(monkeypatch) -> None:
    adapter = _Adapter()
    runtime = _Runtime()
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
    router._agent_require_fresh_evidence = True

    result = router.generate_text(
        "coder",
        [{"role": "user", "content": "Research exact registration APIs and return evidence."}],
    )

    assert result == "finished from model-selected project evidence"
    assert runtime.calls == [
        (
            "generation",
            "search_project_rag",
            {"query": "exact project registration contract"},
        )
    ]
