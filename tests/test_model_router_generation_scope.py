from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters import GenerationResponse, ToolCall
from minecraft_mod_ai.model_router import ModelRouter


class _Registry:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            adapter="llama_cpp",
            exclusive_gpu=True,
            provider="local",
        )

    def load_profile(self, profile: str) -> None:
        assert profile == "test"

    def role(self, profile: str, role: str):
        assert profile == "test"
        assert role == "coder"
        return self.config


class _Runtime:
    def __init__(self, events: list[str], active: list[int]) -> None:
        self.events = events
        self.active = active

    def tool_schemas(self, stage: str):
        assert stage == "generation"
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
        assert stage == "generation"
        assert name == "search_code_rag"
        assert arguments == {"query": "repair the registry"}
        assert self.active[0] == 0
        self.events.append("rag")
        return {"hits": [{"path": "Example.java", "line": 1}]}


class _Adapter:
    def __init__(self, events: list[str], active: list[int]) -> None:
        self.events = events
        self.active = active
        self.calls = 0

    def generate_turn(self, request):
        assert self.active[0] == 1
        self.events.append("decode")
        self.calls += 1
        if self.calls == 1:
            return GenerationResponse(
                tool_calls=(
                    ToolCall(
                        id="rag-1",
                        name="search_code_rag",
                        arguments={"query": "repair the registry"},
                    ),
                )
            )
        return GenerationResponse(content="done")

    def generate(self, request):  # pragma: no cover - tool path must use generate_turn
        raise AssertionError("tool-capable generation must not use generate()")


def test_host_rag_runs_outside_gpu_generation_scope(monkeypatch) -> None:
    events: list[str] = []
    active = [0]
    adapter = _Adapter(events, active)
    runtime = _Runtime(events, active)

    monkeypatch.setattr(
        ModelRouter,
        "_new_text_adapter",
        staticmethod(lambda config, *, role: adapter),
    )

    @contextmanager
    def generation_scope(self, config):
        assert active[0] == 0
        active[0] += 1
        events.append("scope_enter")
        try:
            yield
        finally:
            events.append("scope_exit")
            active[0] -= 1

    monkeypatch.setattr(ModelRouter, "_generation_scope", generation_scope)

    router = ModelRouter(
        profile="test",
        registry=_Registry(),
        agent_tool_runtime_factory=lambda **_: runtime,
    )
    router._agent_require_fresh_evidence = True

    result = router.generate_text(
        "coder",
        [{"role": "user", "content": "repair the registry"}],
    )

    assert result == "done"
    assert active == [0]
    assert events == [
        "scope_enter",
        "decode",
        "scope_exit",
        "rag",
        "scope_enter",
        "decode",
        "scope_exit",
    ]
