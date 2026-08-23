from __future__ import annotations

import json
from types import SimpleNamespace

from minecraft_mod_ai.model_adapters import GenerationRequest, ToolCall
from minecraft_mod_ai.model_router import ModelRouter, _agent_tool_round_limit


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object"},
        },
    }


def test_agent_tool_round_limit_is_unbounded_without_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)

    assert _agent_tool_round_limit() is None


def test_agent_tool_round_limit_honors_only_positive_explicit_value(monkeypatch) -> None:
    monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", "17")
    assert _agent_tool_round_limit() == 17

    for raw in ("", "0", "-1", "not-an-integer"):
        monkeypatch.setenv("MMM_AGENT_TOOL_ROUNDS", raw)
        assert _agent_tool_round_limit() is None


def test_progressing_tool_loop_can_run_beyond_legacy_twelve_round_cutoff(monkeypatch) -> None:
    monkeypatch.delenv("MMM_AGENT_TOOL_ROUNDS", raising=False)
    router = object.__new__(ModelRouter)
    router._agent_require_fresh_evidence = False

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def call(self, stage: str, name: str, arguments: dict[str, object]):
            del stage, name
            step = int(arguments["step"])
            self.calls.append(step)
            return {"step": step}

    class Adapter:
        def __init__(self) -> None:
            self.turns = 0

        def generate_turn(self, request: GenerationRequest):
            self.turns += 1
            if self.turns <= 13:
                arguments = {"step": self.turns}
                return SimpleNamespace(
                    tool_calls=(
                        ToolCall(
                            id=f"call-{self.turns}",
                            name="work_status",
                            arguments=arguments,
                            raw_arguments=json.dumps(arguments),
                        ),
                    ),
                    content="",
                )
            return SimpleNamespace(tool_calls=(), content="done")

    runtime = Runtime()
    adapter = Adapter()
    request = GenerationRequest(
        messages=({"role": "user", "content": "continue until complete"},),
        media_paths=(),
        response_format="text",
        response_schema=None,
        tools=(_tool_schema("work_status"),),
        tool_choice="auto",
        parallel_tool_calls=False,
    )
    config = SimpleNamespace(
        exclusive_gpu=False,
        provider="local",
        adapter="llama_cpp",
    )

    result = router._generate_with_tools(
        config=config,
        adapter=adapter,
        request=request,
        runtime=runtime,
        stage="planning",
        role="planner",
    )

    assert result == "done"
    assert runtime.calls == list(range(1, 14))
