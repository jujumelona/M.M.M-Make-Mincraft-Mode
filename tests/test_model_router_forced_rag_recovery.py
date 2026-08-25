from __future__ import annotations

from types import SimpleNamespace

import pytest

from minecraft_mod_ai.model_adapters import GenerationResponse, ModelConfigurationError, ToolCall
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
        return (
            {
                "type": "function",
                "function": {
                    "name": "search_code_rag",
                    "description": "Search current project code",
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
        return {"hits": [{"path": "src/main/java/Example.java", "line": 7}]}


def _router(monkeypatch, adapter, runtime: _Runtime, tmp_path) -> ModelRouter:
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
    router.bind_agent_workspace(tmp_path, require_fresh_evidence=True)
    return router


def test_premature_coder_final_forces_rag_tool_choice(monkeypatch, tmp_path) -> None:
    class Adapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                assert request.tool_choice == "auto"
                return GenerationResponse(content="premature final")
            if len(self.requests) == 2:
                assert request.tool_choice == {
                    "type": "function",
                    "function": {"name": "search_code_rag"},
                }
                assert request.parallel_tool_calls is False
                return GenerationResponse(
                    tool_calls=(
                        ToolCall(
                            id="forced_rag",
                            name="search_code_rag",
                            arguments={"query": "current registration implementation"},
                            raw_arguments='{"query":"current registration implementation"}',
                        ),
                    )
                )
            assert request.tool_choice == "auto"
            return GenerationResponse(content="grounded final")

    adapter = Adapter()
    runtime = _Runtime()
    router = _router(monkeypatch, adapter, runtime, tmp_path)

    result = router.generate_text(
        "coder",
        [{"role": "user", "content": "Implement the registration change"}],
    )

    assert result == "grounded final"
    assert runtime.calls == [
        (
            "generation",
            "search_code_rag",
            {"query": "current registration implementation"},
        )
    ]
    assert len(adapter.requests) == 3


def test_forced_rag_refusal_is_bounded_and_specific(monkeypatch, tmp_path) -> None:
    class RefusingAdapter:
        def __init__(self) -> None:
            self.requests = []

        def generate_turn(self, request):
            self.requests.append(request)
            if len(self.requests) > 1:
                assert request.tool_choice == {
                    "type": "function",
                    "function": {"name": "search_code_rag"},
                }
                assert request.parallel_tool_calls is False
            return GenerationResponse(content="same premature final")

    adapter = RefusingAdapter()
    runtime = _Runtime()
    router = _router(monkeypatch, adapter, runtime, tmp_path)

    with pytest.raises(ModelConfigurationError, match="Host-selected RAG action"):
        router.generate_text(
            "coder",
            [{"role": "user", "content": "Implement the registration change"}],
        )

    assert len(adapter.requests) == 2
    assert runtime.calls == []
